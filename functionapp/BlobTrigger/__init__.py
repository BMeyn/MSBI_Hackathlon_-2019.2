import logging

import azure.functions as func


def main(InputBlob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {InputBlob.name}\n"
                 f"Blob Size: {InputBlob.length} bytes \n")
                 