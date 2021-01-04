from datetime import datetime
import click
import debie
import json


# Evaluation with config file
from bias_evaluation import evaluation_controller


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
    if 'scores' in config_data:
        debie.scores = config_data['scores']
    else:
        debie.scores = 'all'
    debie.specification_data = config_data
    return bias_evaluation()


def bias_evaluation():
    print("\nDEBIE -- Bias Evaluation with " + debie.scores + " scores started at " + str(datetime.now()))
    bar = {"space": debie.space, "lower": debie.lower, "uploaded": debie.uploaded, "json": debie.json_value}
    scores, used_space, used_lower, not_found, deleted = evaluation_controller.evaluation(debie.scores, debie.specification_data, bar)
    output = debie.scores_to_output(scores, used_space, used_lower, not_found, deleted)
    print(output)
    return output


if __name__ == '__main__':
    configfile()
