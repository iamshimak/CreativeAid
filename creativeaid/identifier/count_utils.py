"""Utilities for extracting probabilities and measures from counts dict"""

import logging
import math

logging.basicConfig(format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.NOTSET)


def retrieve_count(d, verb, noun):
    """Returns count from dict, given a verb and a noun"""
    try:
        return float(d[verb][noun])
    except KeyError:
        return 0


def retrieve_verb_total(d, verb):
    """Returns total count for a verb"""
    try:
        return float(sum(d[verb].values()))
    except KeyError:
        return 0


def retrieve_noun_total(d, noun):
    """Returns total count for a noun"""
    total = 0
    for verb in d:
        try:
            total += d[verb][noun]
        except KeyError:
            pass
    return float(total)


def retrieve_noun_totals(d):
    """Returns dict of all total counts for all nouns"""
    noun_totals = {}
    noun_set = retrieve_noun_set(d)
    for noun in noun_set:  # Initialize dict
        noun_totals[noun] = 0
    for verb in d:
        for noun in d[verb]:
            noun_totals[noun] += d[verb][noun]
    return noun_totals


def retrieve_noun_sum(d):
    """Returns sum of noun counts in the dict"""
    total = 0
    for verb in d:
        total += sum(d[verb].values())
    return float(total)


def retrieve_noun_set(d):
    """Returns set of all nouns in the dict"""
    noun_list = []
    for verb in d:
        noun_list += d[verb].keys()
    return set(noun_list)


def get_conditional_prob(d, verb, noun, verb_total=None):
    """Returns P(noun|verb)"""
    if not verb_total:  # For efficiency
        verb_total = retrieve_verb_total(d, verb)
    try:
        return retrieve_count(d, verb, noun) / verb_total
    except ZeroDivisionError:
        return 0


def get_log_prob(d, verb, noun, verb_total=None):
    """Returns log(P(noun|verb))"""
    if not verb_total:  # For efficiency
        verb_total = retrieve_verb_total(d, verb)
    try:
        return math.log(get_conditional_prob(d, verb, noun, verb_total=verb_total))
    except ValueError:
        return 'None'


def get_noun_prob(noun, noun_totals, noun_sum):
    """Returns P(noun)"""
    return noun_totals[noun] / noun_sum


def get_sps(d, verb, noun_totals=None, noun_sum=None):
    """Calculates and returns SPS(verb)"""
    try:  # If it never occurs in the corpus, return None
        d[verb]
    except KeyError:
        return None
    SPS = 0
    if not noun_sum:
        noun_sum = retrieve_noun_sum(d)
    verb_total = retrieve_verb_total(d, verb)
    if not noun_totals:
        noun_totals = retrieve_noun_totals(d)
    for noun in d[verb].keys():
        if noun:
            prior_prob = get_noun_prob(noun, noun_totals, noun_sum)
            post_prob = get_conditional_prob(d, verb, noun, verb_total=verb_total)
            SPS += post_prob * math.log(post_prob / prior_prob)
    return SPS


def get_sa(d, verb, noun, noun_totals=None, noun_sum=None, sps=None):
    """Calculates and returns SA(verb)"""
    try:  # If it never occurs in the corpus, return None
        d[verb][noun]
    except KeyError:
        return None
    if not noun_sum:
        noun_sum = retrieve_noun_sum(d)
    if not noun_totals:
        noun_totals = retrieve_noun_totals(d)
    SA = 1 / get_sps(d, verb, noun_totals=noun_totals, noun_sum=noun_sum)
    SA *= get_conditional_prob(d, verb, noun)
    SA *= math.log(get_conditional_prob(d, verb, noun) / get_noun_prob(noun, noun_totals, noun_sum))
    return SA


def print_metrics(d, verb, noun, noun_totals=None, noun_sum=None):
    """Returns conditional probability, SPS and SA"""
    if not noun_sum:
        noun_sum = retrieve_noun_sum(d)
    if not noun_totals:
        noun_totals = retrieve_noun_totals(d)
    print('Number of nouns occurring with for verb %s: %d' % (verb, len(d[verb])))
    print('Count of noun %s: %d' % (noun, noun_totals[noun]))
    print('Count of %s and %s: %d' % (verb, noun, retrieve_count(d, verb, noun)))
    print('SPS of %s: %s' % (verb, get_sps(d, verb, noun_totals=noun_totals, noun_sum=noun_sum)))
    print('Conditional probability of %s and %s: %s' % (verb, noun, get_conditional_prob(d, verb, noun)))
    if get_conditional_prob(d, verb, noun) > 0:
        print('Log conditional probability of %s and %s: %s' % (
            verb, noun, math.log(get_conditional_prob(d, verb, noun))))
    if get_sa(d, verb, noun, noun_totals=noun_totals, noun_sum=noun_sum):
        print('SA of %s and %s: %.2fe-03' % (
            verb, noun, 1000 * get_sa(d, verb, noun, noun_totals=noun_totals, noun_sum=noun_sum)))
    print('')


def get_metrics(d, verb, noun, noun_totals=None, noun_sum=None):
    """Returns conditional probability, SPS and SA"""
    if not noun_sum:
        noun_sum = retrieve_noun_sum(d)
    if not noun_totals:
        noun_totals = retrieve_noun_totals(d)
    cond_prob = get_conditional_prob(d, verb, noun)
    log_prob = get_log_prob(d, verb, noun)
    sps = get_sps(d, verb, noun_totals=noun_totals, noun_sum=noun_sum)
    sa = get_sa(d, verb, noun, noun_totals=noun_totals, noun_sum=noun_sum)
    return cond_prob, log_prob, sps, sa
