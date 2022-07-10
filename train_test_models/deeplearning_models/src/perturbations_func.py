import re
import os
import random
import copy
import numpy as np

def SubstituteSimilarSymbolsFunction(test_data, review_num, word_index, words):
    """
    Perturbation which substitutes certain letters with similar symbols

    Params
    test_data: The reviews of big_test_data
    review_num: The chosen review
    word_index: The chosen word that should be changed
    words: List of all the words which will be perturbed (Not used in the fucntion)

    The idea is that the letters 'a', 'i', 'o', 's' are substituted with the letters '@', '1', '0', '$'.
    To make the word easier to read, only one letter is changed in each word.
    """
    word_list = (test_data[review_num]).split()
    # Defining letters which are substitued with the symboly in the symbol list
    letter_list = ['a', 'i', 'o', 's', 'g', 'b', 'z', 'n', 'e', 'u', 'y', 't', 'r', 'j', 'f', 'd',
                   '@', '1', '0', '$', '1', '9', '8', '2', 'm', '€', 'v', 'l']
    symbol_list = ['@', '1', '0', '$', '9', '8', '2', 'm', '€', 'v', 'v', 'l', 'i', 'i', 'l', 'o',
                   'a', 'i', 'o', 's', 'l', 'g', 'b', 'z', 'n', 'e', 'u', 't']

    break_variable=0
    letter_list = np.array(letter_list)
    symbol_list = np.array(symbol_list)
    randomize = np.arange(len(letter_list))
    np.random.shuffle(randomize)
    letter_list = letter_list[randomize]
    symbol_list = symbol_list[randomize]
    for j in range(0, len((letter_list))):
        for i in range(0, len(str(word_list[word_index]))):
            #check whether the letters are one of the ones which should be substituted
            new_wordlist = list(word_list[word_index])
            if new_wordlist[i] == letter_list[j]:
                new_word = list(word_list[word_index])
                new_word[i] = symbol_list[j]
                word_list[word_index] = ''.join(new_word)
                break_variable=1
                break #break j-loop
        if break_variable==1:
            break #break i-loop
    review = ' '.join(word_list)
    return review


def SubstituteSimilarSymbolsFunction_OLD(test_data, review_num, word_index, words):
    """
    Perturbation which substitutes certain letters with similar symbols

    Params
    test_data: The reviews of big_test_data
    review_num: The chosen review
    word_index: The chosen word that should be changed
    words: List of all the words which will be perturbed (Not used in the fucntion)

    The idea is that the letters 'a', 'i', 'o', 's' are substituted with the letters '@', '1', '0', '$'.
    To make the word easier to read, only the one letter is changed in each word.
    """
    word_list = (test_data[review_num]).split()
    # Defining letters which are substitued with the symboly in the symbol list
    letter_list = ['a', 'i', 'o', 's', 'l', 'g', 'b', 'z', 'n', 'e', 'u']
    symbol_list = ['@', '1', '0', '$', '1', '9', '8', '2', 'm', '€', 'v']

    break_variable=0
    for i in range(0, len(str(word_list[word_index]))):
        for j in range(0, len((letter_list))):
            #check whether the letters are one of the ones which should be substituted
            new_wordlist = list(word_list[word_index])
            if new_wordlist[i] == letter_list[j]:
                new_word = list(word_list[word_index])
                new_word[i] = symbol_list[j]
                word_list[word_index] = ''.join(new_word)
                break_variable=1
                break #break j-loop
        if break_variable==1:
            break #break i-loop
    review = ' '.join(word_list)
    return review


def SubstituteNeighborKeyboardFunction(test_data, review_num, word_index, words):
    """
    Perturbation which substitutes certain letters with its neigboring letter on a german keyboard

    Params
    test_data: The reviews of big_test_data
    review_num: The chosen review
    word_index: The chosen word that should be changed
    words: List of all the words which will be perturbed (Not used in the function)

    The idea is that a randomly chosen letter is substituted with the letter which is
    his neighboring letter on a german keyboard.
    """
    # Split data set into list of words
    word_list = (test_data[review_num]).split()

    # build neighbor list for letters
    letters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
               'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
               'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
               'z', 'x', 'c', 'v', 'b', 'n', 'm', 'n']

    letter_neighbors = [['2', 'q'], ['1', '3', 'q', 'w'], ['2', '4', 'w', 'e'],
                        ['3', '5', 'e', 'r'], ['4', '6', 'r', 't'], ['5', '7', 't', 'y'], ['6', '8', 'y', 'u'],
                        ['7', '9', 'u', 'i'], ['8', '0', 'i', 'o'], ['9', 'o', 'p'],
                        ['1', '2', 'w', 'a', 's'], ['2', '3', 'q', 'e', 'a', 's'], ['3', '4', 'w', 'r', 's', 'd'],
                        ['4', '5', 'e', 't', 'd', 'f'], ['5', '6', 'r', 'y', 'f', 'g'],
                        ['6', '7', 't', 'u', 'g', 'h'], ['7', '8', 'y', 'i', 'h', 'j'],
                        ['8', '9', 'u', 'o', 'j', 'k'], ['9', '0', 'i', 'p', 'k', 'l'], ['0', 'o', 'l'],
                        ['q', 'w', 's', 'z'],['a', 'w', 'e', 'd', 'x', 'z'],['s', 'e', 'r', 'f', 'c', 'x'],
                        ['d', 'r', 't', 'g', 'v', 'c'], ['t', 'y', 'f', 'h', 'v', 'b'],
                        ['g', 'y', 'u', 'j', 'n', 'b'], ['h', 'u', 'i', 'k', 'm', 'n'],
                        ['j', 'i', 'o', 'l', 'm'], ['k', 'o', 'p'], ['a', 's', 'x'],
                        ['z', 's', 'd', 'c'], ['x', 'd', 'f', 'v'], ['c', 'f', 'g', 'b'],
                        ['v', 'g', 'h', 'n'], ['b', 'h', 'j', 'm'], ['n', 'j', 'k']
                       ]
    if len(str(word_list[word_index])) >= 2:

        # random number to exchange letter
        x = random.randint(0, len(str(word_list[word_index])) - 1)
        liste = list(word_list[word_index])

        # We check whether the randomly chosen character is in the list
        if (liste[x] not in letters) == False:

            # get index in neighbor list
            letter_index = letters.index(liste[x])
            y = random.randint(0, len(letter_neighbors[letter_index]) - 1)
            liste[x] = letter_neighbors[letter_index][y]

            # Rejoin letters of Word to a proper word
            word_list[word_index] = ''.join(liste)

            review = ' '.join(word_list)
        else:
            review = ' '.join(word_list)
    else:
        review = ' '.join(word_list)
    return review

def SwappingNeighborLetterFunction(test_data, review_num, word_index, words):
    """
    Perturbation which exchanges to neighboring letters

    Params
    test_data: The reviews of big_test_data
    review_num: The chosen review
    word_index: The chosen word that should be changed
    words: List of all the words which will be perturbed (Not used in the function)

    The idea is to choose a letter randomly (but not he last letter) from a word.
    Then this character is exchanged with the next character (on his right).
    """
    # Split data set into list of words
    word_list = (test_data[review_num]).split()
    if len(str(word_list[word_index])) >= 2:

        # random number to exchange letter
        x = random.randint(0, len(str(word_list[word_index])) - 2)

        liste = list(word_list[word_index])
        new_liste = copy.deepcopy(liste)

        #Exchange the character with his right neighbor
        new_liste[x + 1] = liste[x]
        new_liste[x] = liste[x + 1]

        # Rejoin letters of Word to a proper word
        word_list[word_index] = ''.join(new_liste)
        review = ' '.join(word_list)
    else:
        review = ' '.join(word_list)
    return review


def SubstituteNeighborKeyboardFunction_OLD(test_data, review_num, word_index, words):
    """
    Perturbation which substitutes certain letters with its neigboring letter on a german keyboard

    Params
    test_data: The reviews of big_test_data
    review_num: The chosen review
    word_index: The chosen word that should be changed
    words: List of all the words which will be perturbed (Not used in the function)

    The idea is that a randomly chosen letter is substituted with the letter which is
    his neighboring letter on a german keyboard.
    """
    # Split data set into list of words
    word_list = (test_data[review_num]).split()

    # build neighbor list for letters
    letter_neighbors = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'l', 'k', 'j', 'h', 'g', 'f', 'd', 's', 'a',
                        'z', 'x', 'c', 'v', 'b', 'n', 'm', 'n']
    if len(str(word_list[word_index])) >= 2:

        # random number to exchange letter but not the first or last
        x = random.randint(1, len(str(word_list[word_index])) - 1)
        liste = list(word_list[word_index])

        # We check whether the randomly chosen character is in the
        if (liste[x] not in letter_neighbors) == False:

            # get index in neighbor list of
            neighbor_index = letter_neighbors.index(liste[x])
            liste[x] = letter_neighbors[neighbor_index + 1]

            # Rejoin letters of Word to a proper word
            word_list[word_index] = ''.join(liste)

            review = ' '.join(word_list)
        else:
            review = ' '.join(word_list)
    else:
        review = ' '.join(word_list)
    return review

############################################################################################
