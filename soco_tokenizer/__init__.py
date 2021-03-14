import logging

logFormatter = logging.Formatter('%(asctime)s-%(name)s:%(lineno)d - %(levelname)s - %(message)s')
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


__version__ = "0.2.9"
__DOWNLOAD_SERVER__ = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/'
from .logging_handlerr import LoggingHandler
from .SentenceTransformer import SentenceTransformer

