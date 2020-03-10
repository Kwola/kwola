import os

import os.path



def getKwolaUserDataDirectory(subDirName):
    """
    This returns a sub-directory within the kwola user data directory.

    It will ensure the path exists prior to returning
    """

    kwolaDirectory = os.path.join(os.environ['HOME'], ".kwola")

    if not os.path.exists(kwolaDirectory):
        os.mkdir(kwolaDirectory)

    subDirectory = os.path.join(kwolaDirectory, subDirName)
    if not os.path.exists(subDirectory):
        os.mkdir(subDirectory)

    return subDirectory

