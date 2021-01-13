import json
from datetime import datetime
import click
import os

import numpy

""" CLI-TOOL """
os.system("")
print("""   """)
print("\033[96m" + """ WELCOME TO
 _____  ______ ____ _____ ______ 
 |  __ \|  ____|  _ \_   _|  ____|
 | |  | | |__  | |_) || | | |__   
 | |  | |  __| |  _ < | | |  __|  
 | |__| | |____| |_) || |_| |____ 
 |_____/|______|____/_____|______|
""")

print(" DEBiasing embeddings Implicitly and Explicitly ")
print(" An application for debiasing embedding spaces and bias evaluation of explicit and implicit bias specifications"
      + "\033[0m")
print("\n")


""" SHORT MANUAL """
help_string = """ ---- Allowed Input Values: ----

You can restart the input data collection anytime by typing "restart" as input value.

For starting bias evaluation:           "e", "eval", "evaluation"
For starting debiasing:                 "d", "deb", "debias", "debiasing"

For selecting an embedding space:       "f" or "fasttext" to use the fastText space
                                        "g" or "glove" to use the GloVe space
                                        "c" or "cbow" to use the CBOW space
                                        "u" or upload" to use an own embedding space
                                    
For using an own embedding space:       If you are using binary vector data separated into an vec and an vocab file,
                                        enter the path to the vec file first and then the path to the vocab file.
                                        If you are using dictionary-like vector data, enter the path to the file as 
                                        vec file and leave the vocab file blank.
                                        (Important: Enter with the fle endings!)
                                    
Path to bias specification file:        Enter the path to the JSON-file containing the bias specification data.

Lower the sets of the specification:    "t" or "true" or "True" to lower all characters, 
                                        "f", "false", or "False" to leave them as they are.
                                        
Evaluation scores:                      "a" or "all", "e" or "ect", "b" or "bat", "w" or "weat", "k" or "kmeans", 
                                        "s" or "svm", 
                                        "word" or "wordsim" or "wordsim-353, "sim" or "simlex" or "simlex-999"
                                        
Debiasing methods:                       "g" or "gbdd", "b" or "bam", "bamxgbdd", "gbddxbam".

Save complete vector space:             "t" or "true" or "True" to save complete space, "f", "false", or "False" to 
                                        only save the space containing the used terms.

Enable/Disable PCA output:              "t" or "true" or "True" to enable, "f", "false", or "False" to disable pca.

"""


""" IMPORTS """
print("\033[93m" + " --- LOADING DATA --- " + "\033[0m")

import data_controller
import upload_controller

from bias_evaluation import evaluation_controller
from debiasing import debiasing_controller

print("\033[92m" + " --- COMPLETED --- " + "\033[0m")
print("")

print("For help type \"help\"")
print("")

""" Global variables """
method = ""
uploaded = ""
space = ""
json_value = ""
lower = ""
scores = ""
debiasing = ""
pca = ""
vocab_file = ""
vec_file = ""
specification_file = ""
specification_data = {}


@click.command()
@click.option('--selection', prompt="Evaluate a bias specification or debias an embedding space")
def select_method(selection):
    global method
    if selection == "help":
        print(help_string)
        return select_method()
    if selection == "e" or selection == "eval" or selection == "evaluation":
        method = "evaluation"
        if space != "" and specification_file != "":
            return continue_with_prev_input()
        print("")
        return select_space()
    if selection == "d" or selection == "deb" or selection == "debias" or selection == "debiasing":
        method = "debiasing"
        if space != "" and specification_file != "":
            return continue_with_prev_input()
        print("")
        return select_space()
    print("\033[93m" + 'No accepted input value' + "\033[0m" + "\n")
    return select_method()


@click.command()
@click.option('--input', prompt="Continue with current data")
def continue_with_prev_input(input):
    if input == "help":
        print(help_string)
        return continue_with_prev_input()
    if input == "y" or input == "yes" or input == "Yes" or input == "YES":
        if method == "evaluation":
            return select_scores()
        if method == "debiasing":
            return select_debiasing()
    if input == "n" or input == "no" or input == "No" or input == "NO":
        if method == "evaluation":
            return select_space()
        if method == "debiasing":
            return select_space()


@click.command()
@click.option('--input', prompt="Operating Embedding Space")
def select_space(input):
    global space
    global uploaded
    if input == "help":
        print(help_string)
        return select_space()
    if input == "f" or input == "fasttext":
        space = "fasttext"
        return select_specification_file()
    if input == "g" or input == "glove":
        space = "glove"
        return select_specification_file()
    if input == "c" or input == "cbow":
        space = "cbow"
        return select_specification_file()
    if input == "u" or input == "upload":
        uploaded = "true"
        return select_upload()
    if input == "restart":
        return select_method()
    print("\033[93m" + input + ' is no accepted input value for space.' + "\033[0m")
    return select_space()


@click.command()
@click.option("--vector_file", prompt="Vector file")
@click.option("--vocabulary_file", prompt="Vocab file")
def select_upload(vector_file, vocabulary_file=""):
    global vocab_file
    global vec_file
    global space_file
    global space
    if vector_file == "help":
        print(help_string)
        return select_upload()
    if vocab_file == "help":
        print(help_string)
        return select_upload()
    if vector_file == "restart" or vocab_file == "restart":
        return select_method()
    if vector_file != "" and vocabulary_file != "":
        upload_controller.uploaded_binary = 'true'
        vocab_file = vocabulary_file
        vec_file = vector_file
        space = vector_file
        upload_controller.uploaded_vocab, upload_controller.uploaded_vecs = upload_controller.load_binary_embeddings(vocab_file, vec_file)
        return select_specification_file()
    if vector_file != "":
        upload_controller.uploaded_binary = 'false'
        space_file = vector_file
        space = vector_file
        upload_controller.uploaded_space = upload_controller.load_dict_uploaded_file(space_file)
        return select_specification_file()
    print("\033[93m" + " Upload error. Please try again." + "\033[0m")
    return select_space()


@click.command()
@click.option('--input', prompt="Path to bias specification file")
def select_specification_file(input):
    global specification_file
    global specification_data
    if input == "help":
        print(help_string)
        return select_specification_file()
    if input == "restart":
        return select_method()
    try:
        with open(input) as json_file:
            specification_data = json.load(json_file)
            specification_file = input
        return select_lower()
    except FileNotFoundError:
        print("\033[91m" + "Could not find file " + input + " - Please try again." + "\033[0m\n")
        return select_specification_file()


@click.command()
@click.option('--input', prompt="Lower input characters")
def select_lower(input):
    global lower
    global method
    if input == "help":
        print(help_string)
        return select_lower()
    if input == "restart":
        return select_method()
    if input == "f" or input == "false" or input == "False":
        lower = "false"
        if method == "evaluation":
            return select_scores()
        else:
            return select_debiasing()
    if input == "t" or input == "true" or input == "True":
        lower = "true"
        if method == "evaluation":
            return select_scores()
        else:
            return select_debiasing()
    print("\033[93m" + input + ' is no accepted input value for lower.' + "\033[0m")
    return select_lower()


@click.command()
@click.option('--input', prompt="Evaluation scores")
def select_scores(input):
    global scores
    if input == "help":
        print(help_string)
        return select_scores()
    if input == "restart":
        return select_method()
    if input == "a" or input == "all":
        scores = "all"
        return bias_evaluation()
    if input == "e" or input == "ect":
        scores = "ect"
        return bias_evaluation()
    if input == "b" or input == "bat":
        scores = "bat"
        return bias_evaluation()
    if input == "w" or input == "weat" or input == "WEAT":
        scores = "weat"
        return bias_evaluation()
    if input == "k" or input == "kmeans":
        scores = "kmeans"
        return bias_evaluation()
    if input == "s" or input == "svm":
        scores = "svm"
        return bias_evaluation()
    if input == "sim" or input == "simlex" or input == "simlex-999":
        scores = "simlex"
        return bias_evaluation()
    if input == "word" or input == "wordsim" or input == "wordSim-353":
        scores = "wordsim"
        return bias_evaluation()
    print("\033[93m" + input + ' is no accepted input value for scores' + "\033[0m")
    return select_scores()


@click.command()
@click.option('--input', prompt="Debiasing method")
def select_debiasing(input):
    global debiasing
    if input == "help":
        print(help_string)
        return select_debiasing()
    if input == "restart":
        return select_method()
    if input == "BAM" or input == "bam" or input == "b":
        debiasing = "bam"
        return full_debiasing_output()
    if input == "GBDD" or input == "gbdd" or input == "g":
        debiasing = "gbdd"
        return full_debiasing_output()
    if input == "BAMGBDD" or input == "bamgbdd" or input == "bamXgbdd":
        debiasing = "bamXgbdd"
        return full_debiasing_output()
    if input == "GBDDBAM" or input == "gbddbam" or input == "gbddXbam":
        debiasing = "gbddXbam"
        return full_debiasing_output()
    print("\033[93m" + input + ' is no accepted input value as debiasing method' + "\033[0m")
    return select_debiasing()


@click.command()
@click.option('--input', prompt="Save complete vector space")
def full_debiasing_output(input):
    global debiasing

    if input == "help":
        print(help_string)
        return full_debiasing_output()
    if input == "restart":
        return select_method()
    if input == "f" or input == "false" or input == "False":
        return select_pca()
    if input == "t" or input == "true" or input == "True":
        debiasing = "full-" + debiasing
        return save_full_space()
    print("\033[93m" + input + ' is no accepted input value for lower.' + "\033[0m")
    return full_debiasing_output()


def save_full_space():
    global space
    global uploaded
    global specification_data
    global specification_file
    global lower
    global debiasing
    global pca

    print("\nDEBIE -- " + debiasing + "-Debiasing started at " + str(datetime.now()))
    bar = {"space": space, "lower": lower, "pca": pca, "uploaded": uploaded}
    vocab, vecs = debiasing_controller.debiasing(debiasing, specification_data, bar)
    vocab_filename = specification_file.replace(".json", "-debiased.vocab")
    vecs_filename = specification_file.replace(".json", "-debiased.vecs")
    numpy.save(vocab_filename, vocab, allow_pickle=True)
    numpy.save(vecs_filename, vecs, allow_pickle=True)
    print("\n\033[96m" + "DEBIE -- Debiased Space saved as " + vocab_filename + " and  " + vecs_filename + "\033[0m")
    print("    Time: " + str(datetime.now()) + "\n")


@click.command()
@click.option('--input', prompt="Enable PCA")
def select_pca(input):
    global pca
    if input == "help":
        print(help_string)
        return select_pca
    if input == "restart":
        return select_method()
    if input == "f" or input == "false" or input == "False":
        pca = "false"
        return exe_debiasing()
    if input == "t" or input == "true" or input == "True":
        pca = "true"
        return exe_debiasing()
    print("\033[93m" + input + ' is no accepted input value for PCA.' + "\033[0m")
    return select_pca()


def bias_evaluation():
    global space
    global uploaded
    global specification_data
    global lower
    global scores
    global json_value
    print("\nDEBIE -- Bias Evaluation with " + scores + " scores started at " + str(datetime.now()))
    bar = {"space": space, "lower": lower, "uploaded": uploaded, "json": json_value}
    scores, used_space, used_lower, not_found, deleted = evaluation_controller.evaluation(scores, specification_data, bar)
    output = scores_to_output(scores, used_space, used_lower, not_found, deleted)
    print(output)
    return select_method()


def scores_to_output(score_dict, used_space, used_lower, not_found, deleted):
    output = ""
    output += "\n\033[96m" + "DEBIE -- Bias Evaluation Scores:" + "\033[0m" + "\n"
    if "ECT_Score" in score_dict:
        output += "    ECT-Score:           " + str(score_dict["ECT_Score"]) + "\n"
    if "ECT_P_Value" in score_dict:
        output += "    ECT P-Value:         " + str(score_dict["ECT_P_Value"]) + "\n"
    if "BAT_Score" in score_dict:
        output += "    BAT-Score:           " + str(score_dict["BAT_Score"]) + "\n"
    if "WEAT_Effect_Size" in score_dict:
        output += "    WEAT Effect-Size:    " + str(score_dict["WEAT_Effect_Size"]) + "\n"
    if "WEAT_P_Value" in score_dict:
        output += "    WEAT_P_Value:        " + str(score_dict["WEAT_P_Value"]) + "\n"
    if "K_Means" in score_dict:
        output += "    K_Means:             " + str(score_dict["K_Means"]) + "\n"
    if "SVM" in score_dict:
        output += "    SVM:                 " + str(score_dict["SVM"]) + "\n"
    if "SimLexPearson" in score_dict:
        output += "    SimLexPearson:       " + str(score_dict["SimLexPearson"]) + "\n"
    if "SimLexSpearman" in score_dict:
        output += "    SimLexSpearman:      " + str(score_dict["SimLexSpearman"]) + "\n"
    if "WordSimPearson" in score_dict:
        output += "    WordSimPearson:      " + str(score_dict["WordSimPearson"]) + "\n"
    if "WordSimSpearman" in score_dict:
        output += "    WordSimSpearman:     " + str(score_dict["WordSimSpearman"]) + "\n"

    output += "\n" + "    Used embedding space: " + used_space + "\n"
    output += "    Lowered input values: " + used_lower + "\n"
    output += "    Not in embedding space found words: " + str(not_found) + "\n"
    output += "    Deleted words (due to unequal set lengths): " + str(deleted) + "\n"
    output += "    Time: " + str(datetime.now()) + "\n"

    return output


def exe_debiasing():
    global space
    global uploaded
    global specification_data
    global specification_file
    global lower
    global debiasing
    global pca

    print("\nDEBIE -- " + debiasing + "-Debiasing started at " + str(datetime.now()))
    bar = {"space": space, "lower": lower, "pca": pca, "uploaded": uploaded}
    response, status_code = debiasing_controller.debiasing(debiasing, specification_data, bar)
    filename = specification_file.replace(".json", "").replace(".txt", "") + "-debiased" + ".json"
    file = open(filename, "w")
    file.write(response)
    print("\n\033[96m" + "DEBIE -- Debiased Space saved as " + filename + "\033[0m")
    print("    Time: " + str(datetime.now()) + "\n")

    return select_method()


@click.command()
@click.option('--input', prompt='Evaluate test set on debiased space')
def continue_evaluation(input):
    global space
    global json_value
    if input == "help":
        print(help_string)
        return select_pca
    if input == "restart":
        return select_method()
    if input == "f" or input == "false" or input == "False":
        return select_method()
    if input == "t" or input == "true" or input == "True":
        space = 'uploaded'
        json_value = 'true'
        return bias_evaluation()
    print("\033[93m" + input + ' is no accepted input value for PCA.' + "\033[0m")
    return select_pca()


def bias_evaluation_config(configfile):
    global space
    global uploaded
    global json_value
    global lower
    global scores
    global specification_data

    with open(configfile) as json_file:
        config_data = json.load(json_file)
    if 'space' in config_data:
        space = config_data['space']
    else:
        space = 'fasttext'
    if 'uploaded' in config_data:
        uploaded = config_data['uploaded']
    else:
        uploaded = "false"
    if 'json' in config_data:
        json_value = config_data['json']
    if 'lower' in config_data:
        lower = config_data['lower']
    if 'scores' in config_data:
        scores = config_data['scores']
    else:
        scores = 'all'
    specification_data = config_data

    print("\nDEBIE -- Bias Evaluation with " + scores + " scores started at " + str(datetime.now()))
    bar = {"space": space, "lower": lower, "uploaded": uploaded, "json": json_value}
    scores, used_space, used_lower, not_found, deleted = evaluation_controller.evaluation(scores, specification_data, bar)
    output = scores_to_output(scores, used_space, used_lower, not_found, deleted)
    print(output)
    return output


def debiasing_config(configfile):
    global space
    global uploaded
    global json_value
    global lower
    global debiasing
    global specification_data
    global specification_file

    with open(configfile) as json_file:
        config_data = json.load(json_file)

    if 'space' in config_data:
        space = config_data['space']
    else:
        space = 'fasttext'
    if 'uploaded' in config_data:
        uploaded = config_data['uploaded']
    else:
        uploaded = "false"
    if 'json' in config_data:
        json_value = config_data['json']
    if 'lower' in config_data:
        lower = config_data['lower']
    if 'debiasing' in config_data:
        debiasing = config_data['debiasing']
    else:
        debiasing = 'bam'
    if 'pca' in config_data:
        pca = config_data['pca']
    specification_data = config_data
    specification_file = configfile

    print("\nDEBIE -- " + debiasing + "-Debiasing started at " + str(datetime.now()))
    bar = {"space": space, "lower": lower, "pca": pca, "uploaded": uploaded}
    response, status_code = debiasing_controller.debiasing(debiasing, specification_data, bar)
    filename = specification_file.replace(".json", "").replace(".txt", "") + "-debiased" + ".json"
    file = open(filename, "w")
    file.write(response)
    print("\n\033[96m" + "DEBIE -- Debiased Space saved as " + filename + "\033[0m")
    print("    Time: " + str(datetime.now()) + "\n")

    return file


@click.command()
@click.option('--mode', type=str)
@click.option('--config', type=click.Path(exists=True))
def debie(mode=None, config=None):
    if mode == 'evaluation':
        return bias_evaluation_config(config)
    if mode == 'debiasing':
        return debiasing_config(config)
    return select_method()


if __name__ == '__main__':
    debie()
