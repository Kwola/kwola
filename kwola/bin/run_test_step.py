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


from ..config.config import Configuration
from ..datamodels.CustomIDField import CustomIDField
from ..datamodels.TestingStepModel import TestingStep
from ..tasks import RunTestingStep
from .main import getConfigurationDirFromCommandLineArgs


def main():
    """
        This is the entry point for the Kwola secondary command, kwola_run_test_step.
    """
    configDir = getConfigurationDirFromCommandLineArgs()
    config = Configuration(configDir)

    testingStep = TestingStep(id=CustomIDField.generateNewUUID(TestingStep, config))
    testingStep.saveToDisk(config)

    RunTestingStep.runTestingStep(configDir, str(testingStep.id))
