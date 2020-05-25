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


from .errors.BaseError import BaseError
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *


class BugModel(Document):
    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    testingStepId = StringField()

    testingRunId = StringField(required=False)

    executionSessionId = StringField()

    stepNumber = IntField()

    error = EmbeddedDocumentField(BaseError)

    reproductionTraces = ListField(StringField())

    def saveToDisk(self, config, subFolderStr, overrideSaveFormat=None, overrideCompression=None):
        saveObjectToDisk(self, subFolderStr, config, overrideSaveFormat=overrideSaveFormat, overrideCompression=overrideCompression)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):

        return loadObjectFromDisk(BugModel, id, "bugs", config, printErrorOnFailure=printErrorOnFailure)

    def loadBugFromDisk(id, config, subFolder, printErrorOnFailure=True):

        return loadObjectFromDisk(BugModel, id, subFolder, config, printErrorOnFailure=printErrorOnFailure)
            
    def generateBugText(self):
        return self.error.generateErrorDescription()