from datetime import datetime

import debie
import click
import json

# Evaluation with config file
from debiasing_models import debiasing_controller


@click.command()
@click.argument('configfile')
def configfile(configfile):

    with open(configfile) as json_file:
        config_data = json.load(json_file)

    if 'space' in config_data:
        debie.space = config_data['space']
    else:
        debie.space = 'fasttext'
    if 'uploaded' in config_data:
        debie.uploaded = config_data['uploaded']
    else:
        debie.uploaded = "false"
    if 'json' in config_data:
        debie.json_value = config_data['json']
    if 'lower' in config_data:
        debie.lower = config_data['lower']
    if 'debiasing_models' in config_data:
        debie.debiasing = config_data['debiasing_models']
    else:
        debie.debiasing = 'bam'
    if 'pca' in config_data:
        debie.pca = config_data['pca']
    debie.specification_data = config_data
    debie.specification_file = configfile
    return exe_debiasing()


def exe_debiasing():

    print("\nDEBIE -- " + debie.debiasing + "-Debiasing started at " + str(datetime.now()))
    bar = {"space": debie.space, "lower": debie.lower, "pca": debie.pca, "uploaded": debie.uploaded}
    response, status_code = debiasing_controller.debiasing(debie.debiasing, debie.specification_data, bar)
    filename = debie.specification_file.replace(".json", "").replace(".txt", "") + "-debiased" + ".json"
    file = open(filename, "w")
    file.write(response)
    print("\n\033[96m" + "DEBIE -- Debiased Space saved as " + filename + "\033[0m")
    print("    Time: " + str(datetime.now()) + "\n")

    return file


if __name__ == '__main__':
    configfile()
