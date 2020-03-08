from .BaseAgent import BaseAgent
import math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import scipy.signal
import pandas
import cv2
from kwola.models.actions.ClickTapAction import ClickTapAction
import os.path
import numpy
import skimage
import skimage.transform
import skimage.color
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.actions.RightClickAction import RightClickAction
from kwola.models.actions.TypeAction import TypeAction
from kwola.models.actions.WaitAction import WaitAction
import itertools

from .BradNet import BradNet

# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs) if USE_CUDA else autograd.Variable(*args, **kwargs)

Variable = lambda x:x.cuda()

def grouper(n, iterable):
    """Chunks an iterable into sublists"""
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

class DeepLearningAgent(BaseAgent):
    """
        This class represents a deep learning agent, which uses reinforcement learning to make the automated testing more effective
    """

    def __init__(self):
        super().__init__()

        self.num_frames = 1400000
        self.batch_size = 1
        self.gamma = 0.50
        self.frameStart = 7200

    def load(self):
        """
            Loads the agent from db / disk

            :return:
        """

        if os.path.exists("/home/bradley/kwola/deep_learning_model"):
            self.model.load_state_dict(torch.load("/home/bradley/kwola/deep_learning_model"))

    def save(self):
        """
            Saves the agent to the db / disk.

            :return:
        """
        torch.save(self.model.state_dict(), "/home/bradley/kwola/deep_learning_model")


    def initialize(self, environment):
        """
        Initialize the agent for operating in the given environment.

        :param environment:
        :return:
        """
        self.environment = environment

        rect = environment.screenshotSize()

        self.model = BradNet(self.environment.branchFeatureSize() * 2, len(self.actions), self.environment.branchFeatureSize())

        self.model = self.model.cuda()
        # self.model = self.model

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def getImage(self):
        image = cv2.imdecode(numpy.frombuffer(self.environment.driver.get_screenshot_as_png(), numpy.uint8), -1)

        width = image.shape[0]
        height = image.shape[1]

        # shrunk = skimage.transform.resize(image, [int(width / 2), int(height / 2)])

        # Convert to HSL representation, but discard the saturation layer
        image = skimage.color.rgb2hsv(image[:, :, :3])

        swapped = numpy.swapaxes(numpy.swapaxes(image, 0, 2), 1, 2)

        hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)

        return hueLightnessImage


    def getAdditionalFeatures(self):
        branchFeature = numpy.minimum(self.environment.lastCumulativeBranchExecutionVector, numpy.ones_like(self.environment.lastCumulativeBranchExecutionVector))
        decayingExecutionTraceFeature = self.environment.decayingExecutionTrace
        return numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)


    def nextBestAction(self):
        """
            Return the next best action predicted by the agent.
            # :param environment:
            :return:
        """
        epsilon = 0.5
        image = self.getImage()
        additionalFeatures = self.getAdditionalFeatures()

        width = image.shape[2]
        height = image.shape[1]

        if random.random() > epsilon:
            image = Variable(torch.FloatTensor(numpy.array(image)).unsqueeze(0))

            q_value, predictedTrace = self.model({"image": image, "additionalFeature": Variable(torch.FloatTensor(additionalFeatures))})

            action_index = q_value.reshape([-1, width * height * len(self.actionsSorted)]).argmax(1).data[0]

            actionInfo = self.model.actionIndexToActionDetails(width, height, action_index)
        else:

            action_type = random.randrange(len(self.actionsSorted))
            action_x = random.randrange(width)
            action_y = random.randrange(height)

            actionInfo = (action_type, action_x, action_y)
            predictedTrace = None

        action = self.actions[self.actionsSorted[actionInfo[0]]](actionInfo[1], actionInfo[2])

        return action


    def learnFromTestingSequence(self, testingSequence):
        """
            Runs the backward pass / gradient update so the algorithm can learn from all the memories in the given testing sequence

            :param testingSequence:
            :return:
        """

        frames = []

        cap = cv2.VideoCapture(f'/home/bradley/{str(testingSequence.id)}.mp4')

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                image = skimage.color.rgb2hsv(frame[:, :, :3])

                swapped = numpy.swapaxes(numpy.swapaxes(image, 0, 2), 1, 2)

                hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)

                frames.append(hueLightnessImage)
            else:
                break

        # First compute the present reward at each time step
        present_rewards = []
        for trace in testingSequence.executionTraces:
            if trace.didActionSucceed == False:
                present_rewards.append(-0.1)
            elif trace.didNewBranchesExecute or trace.hadNewNetworkTraffic or trace.isURLNew or trace.isScreenshotNew:
                present_rewards.append(0.2)
            else:
                present_rewards.append(0.0)

        discountRate = 0.95

        # Now compute the discounted reward
        discountedRewards = []
        present_rewards.reverse()
        current = 0
        for reward in present_rewards:
            current *= discountRate
            current += reward
            discountedRewards.append(current)

        discountedRewards.reverse()
        present_rewards.reverse()

        shuffledTraceFrameList = list(zip(testingSequence.executionTraces, frames, discountedRewards))
        random.shuffle(shuffledTraceFrameList)

        rewardLosses = []
        tracePredictionLosses = []
        totalLosses = []
        width = None
        height = None

        for batch in grouper(self.batch_size, shuffledTraceFrameList):
            batchFrames = []
            batchAdditionalFeatures = []
            batchActionTypes = []
            batchActionXs = []
            batchActionYs = []
            batchActionIndexes = []
            batchExecutionTraces = []
            batchDiscountedRewards = []

            for trace, frame, discountedReward in batch:
                width = frame.shape[2]
                height = frame.shape[1]

                branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
                decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
                additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

                batchFrames.append(frame)
                batchAdditionalFeatures.append(additionalFeature)

                action_index = self.model.actionDetailsToActionIndex(width, height, self.actionsSorted.index(trace.actionPerformed.type), trace.actionPerformed.x, trace.actionPerformed.y)

                batchActionTypes.append(trace.actionPerformed.type)
                batchActionXs.append(trace.actionPerformed.x)
                batchActionYs.append(trace.actionPerformed.y)
                batchActionIndexes.append(action_index)

                executionTrace = numpy.array(trace.branchExecutionTrace)
                executionTrace = numpy.minimum(executionTrace, numpy.ones_like(executionTrace))

                batchExecutionTraces.append(executionTrace)

                batchDiscountedRewards.append(discountedReward)

            q_values, predictedTraces = self.model({
                "image": Variable(torch.FloatTensor(numpy.array(batchFrames))),
                "additionalFeature": Variable(torch.FloatTensor(batchAdditionalFeatures)),
                "action_type": batchActionTypes,
                "action_x": batchActionXs,
                "action_y": batchActionYs
            })

            q_values = q_values.reshape([-1, height * width * len(self.actions)])

            action_indexes = Variable(torch.LongTensor(numpy.array(batchActionIndexes, dtype=numpy.int32)))
            action_indexes = action_indexes.unsqueeze(1)

            q_value = q_values.gather(1, action_indexes).squeeze(1)

            print(batchDiscountedRewards)
            torchBatchDiscountedRewards = Variable(torch.FloatTensor(numpy.array(batchDiscountedRewards) ))
            #
            print(q_value)
            print(q_value.shape)
            print(torchBatchDiscountedRewards)
            print(torchBatchDiscountedRewards.shape)

            rewardLoss = (q_value - torchBatchDiscountedRewards).pow(2).mean()

            tracePredictionLoss = (predictedTraces - Variable(torch.FloatTensor(batchExecutionTraces))).pow(2).mean()

            totalLoss = rewardLoss + tracePredictionLoss

            self.optimizer.zero_grad()
            totalLoss.backward()
            self.optimizer.step()

            rewardLoss = float(rewardLoss.data.item())
            tracePredictionLoss = float(tracePredictionLoss.data.item())
            totalLoss = float(totalLoss.data.item())

            rewardLosses.append(rewardLoss)
            tracePredictionLosses.append(tracePredictionLoss)
            totalLosses.append(totalLoss)

        averageRewardLoss = numpy.mean(rewardLosses)
        averageTracePredictionLoss = numpy.mean(tracePredictionLosses)
        averageTotalLoss = numpy.mean(totalLosses)

        totalReward = float(numpy.sum(present_rewards))

        print(testingSequence.id)
        print("Total Reward", float(totalReward))
        print("Average Reward Loss:", averageRewardLoss)
        print("Average Trace Predicton Loss:", averageTracePredictionLoss)
        print("Average Total Loss:", averageTotalLoss)
