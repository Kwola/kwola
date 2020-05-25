#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


from ..config.logger import getLogger, setupLocalLogging
from ..components.agents.DeepLearningAgent import DeepLearningAgent
from ..components.environments.WebEnvironment import WebEnvironment
from ..tasks.TaskProcess import TaskProcess
from ..config.config import Configuration
from ..datamodels.ExecutionSessionModel import ExecutionSession
from ..datamodels.BugModel import BugModel
from ..datamodels.CustomIDField import CustomIDField
from ..datamodels.TestingStepModel import TestingStep
from .RunTrainingStep import addExecutionSessionToSampleCache
from datetime import datetime
import atexit
import concurrent.futures
import billiard as multiprocessing
import numpy
import os
import pickle
import tempfile
import time
import traceback


def predictedActionSubProcess(configDir, shouldBeRandom, subProcessCommandQueue, subProcessResultQueue):
    config = Configuration(configDir)

    agent = DeepLearningAgent(config, whichGpu=None)

    agent.initialize(enableTraining=False)
    agent.load()

    while True:
        message = subProcessCommandQueue.get()

        if message == "quit":
            break
        else:
            inferenceBatchFileName = message

        with open(inferenceBatchFileName, 'rb') as file:
            step, images, envActionMaps, pastExecutionTraces = pickle.load(file)

        os.unlink(inferenceBatchFileName)

        actions = agent.nextBestActions(step, images, envActionMaps, pastExecutionTraces, shouldBeRandom=shouldBeRandom)

        resultFileDescriptor, resultFileName = tempfile.mkstemp()
        with open(resultFileDescriptor, 'wb') as file:
            pickle.dump(actions, file)

        subProcessResultQueue.put(resultFileName)


def createDebugVideoSubProcess(configDir, executionSessionId, name="", includeNeuralNetworkCharts=True, includeNetPresentRewardChart=True, hilightStepNumber=None, folder="debug_videos"):
    config = Configuration(configDir)

    agent = DeepLearningAgent(config, whichGpu=None)
    agent.initialize(enableTraining=False)
    agent.load()

    kwolaDebugVideoDirectory = config.getKwolaUserDataDirectory(folder)

    executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

    videoData = agent.createDebugVideoForExecutionSession(executionSession, includeNeuralNetworkCharts=includeNeuralNetworkCharts, includeNetPresentRewardChart=includeNetPresentRewardChart, hilightStepNumber=hilightStepNumber)
    with open(os.path.join(kwolaDebugVideoDirectory, f'{name + "_" if name else ""}{str(executionSession.id)}.mp4'), "wb") as cloneFile:
        cloneFile.write(videoData)

    del agent


def loadAllBugs(config):
    bugsDir = config.getKwolaUserDataDirectory("bugs")

    bugs = []

    for errorFolder in os.listdir(bugsDir):
        for fileName in os.listdir(os.path.join(bugsDir,errorFolder)):
            if ".lock" not in fileName and ".txt" not in fileName and ".mp4" not in fileName:
                bugId = fileName
                bugId = bugId.replace(".json", "")
                bugId = bugId.replace(".gz", "")
                bugId = bugId.replace(".pickle", "")
                subFolder = "bugs/" + errorFolder
                bug = BugModel.loadBugFromDisk(bugId, config, subFolder)

                if bug is not None:
                    bugs.append(bug)

    return bugs


def runAndJoinSubprocess(debugVideoSubprocess):
    debugVideoSubprocess.start()
    debugVideoSubprocess.join()


def runTestingStep(configDir, testingStepId, shouldBeRandom=False, generateDebugVideo=False):
    getLogger().info(f"[{os.getpid()}] Starting New Testing Sequence")

    returnValue = {'success': True}

    try:
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

        config = Configuration(configDir)

        environment = WebEnvironment(config=config)

        stepsRemaining = int(config['testing_sequence_length'])

        testStep = TestingStep.loadFromDisk(testingStepId, config)

        testStep.startTime = datetime.now()
        testStep.status = "running"
        testStep.saveToDisk(config)

        returnValue["testingStepId"] = str(testStep.id)

        executionSessions = [
            ExecutionSession(
                id=str(testingStepId) + "_session_" + str(sessionN),
                owner=testStep.owner,
                testingStepId=str(testingStepId),
                testingRunId=testStep.testingRunId,
                applicationId=testStep.applicationId,
                startTime=datetime.now(),
                endTime=None,
                tabNumber=sessionN,
                executionTraces=[]
            )
            for sessionN in range(environment.numberParallelSessions())
        ]

        executionSessionTraces = [[] for sessionN in range(environment.numberParallelSessions())]

        for session in executionSessions:
            session.saveToDisk(config)

        allKnownErrorHashes = set()
        newErrorsThisTestingStep = []
        newErrorOriginalExecutionSessionIds = []
        newErrorOriginalStepNumbers = []

        bugs = loadAllBugs(config)
        for bug in bugs:
            hash = bug.error.computeHash()
            allKnownErrorHashes.add(hash)

        step = 0

        subProcesses = []

        for n in range(config['testing_subprocess_pool_size']):
            subProcessCommandQueue = multiprocessing.Queue()
            subProcessResultQueue = multiprocessing.Queue()
            subProcess = multiprocessing.Process(target=predictedActionSubProcess, args=(configDir, shouldBeRandom, subProcessCommandQueue, subProcessResultQueue))
            subProcess.start()
            atexit.register(lambda: subProcess.terminate())

            subProcesses.append((subProcessCommandQueue, subProcessResultQueue, subProcess))

        while stepsRemaining > 0:
            stepsRemaining -= 1

            images = environment.getImages()
            envActionMaps = environment.getActionMaps()

            fileDescriptor, inferenceBatchFileName = tempfile.mkstemp()

            with open(fileDescriptor, 'wb') as file:
                pickle.dump((step, images, envActionMaps, executionSessionTraces), file)

            del images, envActionMaps

            subProcessCommandQueue, subProcessResultQueue, subProcess = subProcesses[0]

            subProcessCommandQueue.put(inferenceBatchFileName)
            resultFileName = subProcessResultQueue.get()
            with open(resultFileName, 'rb') as file:
                actions = pickle.load(file)
            os.unlink(resultFileName)

            if stepsRemaining % config['testing_print_every'] == 0:
                getLogger().info(f"[{os.getpid()}] Finished {step + 1} testing actions.")

            traces = environment.runActions(actions, [executionSession.id for executionSession in executionSessions])
            for sessionN, executionSession, trace in zip(range(len(traces)), executionSessions, traces):
                trace.executionSessionId = str(executionSession.id)
                trace.testingStepId = str(testingStepId)
                trace.applicationId = str(executionSession.applicationId)
                trace.testingRunId = str(executionSession.testingRunId)
                trace.owner = testStep.owner
                trace.saveToDisk(config)

                executionSessions[sessionN].executionTraces.append(str(trace.id))
                executionSessionTraces[sessionN].append(trace)
                executionSessions[sessionN].totalReward = float(numpy.sum(DeepLearningAgent.computePresentRewards(executionSessionTraces[sessionN], config)))

                for error in trace.errorsDetected:
                    hash = error.computeHash()

                    if hash not in allKnownErrorHashes:
                        allKnownErrorHashes.add(hash)
                        newErrorsThisTestingStep.append(error)
                        newErrorOriginalExecutionSessionIds.append(str(executionSession.id))
                        newErrorOriginalStepNumbers.append(step)

            if config['testing_reset_agent_period'] == 1 or stepsRemaining % config['testing_reset_agent_period'] == (config['testing_reset_agent_period'] - 1):
                subProcessCommandQueue, subProcessResultQueue, subProcess = subProcesses.pop(0)

                subProcessCommandQueue.put("quit")
                subProcess.terminate()

                subProcessCommandQueue = multiprocessing.Queue()
                subProcessResultQueue = multiprocessing.Queue()
                subProcess = multiprocessing.Process(target=predictedActionSubProcess, args=(configDir, shouldBeRandom, subProcessCommandQueue, subProcessResultQueue))
                subProcess.start()
                atexit.register(lambda: subProcess.terminate())

                subProcesses.append((subProcessCommandQueue, subProcessResultQueue, subProcess))

            step += 1
            del traces

        for subProcessCommandQueue, subProcessResultQueue, subProcess in subProcesses:
            subProcessCommandQueue.put("quit")
            subProcess.join()

        getLogger().info(f"[{os.getpid()}] Creating movies for the execution sessions of this testing sequence.")
        videoPaths = environment.createMovies()

        kwolaVideoDirectory = config.getKwolaUserDataDirectory("videos")

        for sessionN, videoPath, executionSession in zip(range(len(videoPaths)), videoPaths, executionSessions):
            with open(videoPath, 'rb') as origFile:
                with open(os.path.join(kwolaVideoDirectory, f'{str(executionSession.id)}.mp4'), "wb") as cloneFile:
                    cloneFile.write(origFile.read())

        totalRewards = []
        for session in executionSessions:
            getLogger().info(f"[{os.getpid()}] Session {session.tabNumber} finished with total reward: {session.totalReward:.3f}")
            session.saveToDisk(config)
            totalRewards.append(session.totalReward)

        getLogger().info(f"[{os.getpid()}] Mean total reward of all sessions: {numpy.mean(totalRewards):.3f}")

        testStep.bugsFound = len(newErrorsThisTestingStep)
        testStep.errors = newErrorsThisTestingStep

        debugVideoSubprocesses = []

        for session in executionSessions:
            debugVideoSubprocess = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, str(session.id), "", False, False, None, "annotated_videos"))
            atexit.register(lambda: debugVideoSubprocess.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess)

        getLogger().info(f"[{os.getpid()}] Found {len(newErrorsThisTestingStep)} new unique errors this session.")
        for errorIndex, error, executionSessionId, stepNumber in zip(range(len(newErrorsThisTestingStep)), newErrorsThisTestingStep, newErrorOriginalExecutionSessionIds, newErrorOriginalStepNumbers):
            bug = BugModel()
            bug.id = CustomIDField.generateNewUUID(BugModel, config)
            bug.owner = testStep.owner
            bug.applicationId = testStep.applicationId
            bug.testingStepId = testStep.id
            bug.executionSessionId = executionSessionId
            bug.stepNumber = stepNumber
            bug.error = error
            bug.testingRunId = testStep.testingRunId

            #created segmented bug type path
            errDir = bug.error._cls
            subFolderStr = "bugs/" + errDir
            subFolder = os.path.join(config.getKwolaUserDataDirectory("bugs"), errDir) 
            if not os.path.exists(subFolder):
                getLogger().info(f"\n\n[{os.getpid()}] No folder for bug type : {subFolder}, creating...") 
                try:
                    os.mkdir(subFolder)
                except FileExistsError:
                    getLogger().info(f"\n\n[{os.getpid()}] FileExistsError")
            else:
                getLogger().info(f"\n\n[{os.getpid()}] placed in folder for bugs/{bug.error._cls}") 
            #end pass segments to disksave

            bug.saveToDisk(config, subFolderStr, overrideSaveFormat="json", overrideCompression=0)
            bug.saveToDisk(config, subFolderStr)


            bugTextFile = os.path.join(subFolder, bug.id + ".txt")
            with open(bugTextFile, "wt") as file:
                file.write(bug.generateBugText())

            bugVideoFilePath = os.path.join(subFolder, bug.id + ".mp4")
            with open(os.path.join(kwolaVideoDirectory, f'{str(executionSessionId)}.mp4'), "rb") as origFile:
                with open(bugVideoFilePath, 'wb') as cloneFile:
                    cloneFile.write(origFile.read())

            debugVideoSubprocess = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, str(executionSessionId), f"{bug.id}_bug", False, False, stepNumber, subFolderStr))
            atexit.register(lambda: debugVideoSubprocess.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess)

            getLogger().info(f"\n\n[{os.getpid()}] Bug #{errorIndex + 1}:\n{bug.generateBugText()}\n")


        if not shouldBeRandom and generateDebugVideo:
            # Start some parallel processes generating debug videos.
            debugVideoSubprocess1 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, str(executionSessions[0].id), "prediction", True, True, None, "debug_videos"))
            atexit.register(lambda: debugVideoSubprocess1.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess1)

            # Leave a gap between the two to reduce collision
            time.sleep(5)

            debugVideoSubprocess2 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, str(executionSessions[int(len(executionSessions) / 3)].id), "mix", True, True, None, "debug_videos"))
            atexit.register(lambda: debugVideoSubprocess2.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess2)

        environment.shutdown()

        del environment

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for debugVideoSubprocess in debugVideoSubprocesses:
                futures.append(executor.submit(runAndJoinSubprocess, debugVideoSubprocess))
            for future in futures:
                future.result()

        for session in executionSessions:
            getLogger().info(f"[{os.getpid()}] Preparing samples for {session.id} and adding them to the sample cache.")
            addExecutionSessionToSampleCache(session.id, config)

        testStep.status = "completed"
        testStep.endTime = datetime.now()
        testStep.executionSessions = [session.id for session in executionSessions]
        testStep.saveToDisk(config)

    except Exception as e:
        getLogger().error(f"[{os.getpid()}] Unhandled exception occurred during testing sequence:\n{traceback.format_exc()}")
        returnValue['success'] = False
        returnValue['exception'] = traceback.format_exc()

    # This print statement will trigger the parent manager process to kill this process.
    getLogger().info(f"[{os.getpid()}] Finished Running Testing Sequence!")

    return returnValue


if __name__ == "__main__":
    task = TaskProcess(runTestingStep)
    task.run()
