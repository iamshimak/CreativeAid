"""Utilities for extracting probabilities and measures from counts dict"""

import logging
import math


class WordFrequency(object):

    def __init__(self, frequency_model):
        self.d = frequency_model

    def retrieve_count(self, verb, noun):
        """Returns count from dict, given a verb and a noun"""
        try:
            return float(self.d[verb][noun])
        except KeyError:
            return 0

    def retrieve_verb_total(self, verb):
        """Returns total count for a verb"""
        try:
            return float(sum(self.d[verb].values()))
        except KeyError:
            return 0

    def retrieve_noun_total(self, noun):
        """Returns total count for a noun"""
        total = 0
        for verb in self.d:
            try:
                total += self.d[verb][noun]
            except KeyError:
                pass
        return float(total)

    def retrieve_noun_totals(self):
        """Returns dict of all total counts for all nouns"""
        noun_totals = {}
        noun_set = self.retrieve_noun_set()
        for noun in noun_set:  # Initialize dict
            noun_totals[noun] = 0
        for verb in self.d:
            for noun in self.d[verb]:
                noun_totals[noun] += self.d[verb][noun]
        return noun_totals

    def retrieve_noun_sum(self):
        """Returns sum of noun counts in the dict"""
        total = 0
        for verb in self.d:
            total += sum(self.d[verb].values())
        return float(total)

    def retrieve_noun_set(self):
        """Returns set of all nouns in the dict"""
        noun_list = []
        for verb in self.d:
            noun_list += self.d[verb].keys()
        return set(noun_list)

    def get_conditional_prob(self, verb, noun, verb_total=None):
        """Returns P(noun|verb)"""
        if not verb_total:  # For efficiency
            verb_total = self.retrieve_verb_total(verb)
        try:
            return self.retrieve_count(verb, noun) / verb_total
        except ZeroDivisionError:
            return 0

    def get_log_prob(self, verb, noun, verb_total=None):
        """Returns log(P(noun|verb))"""
        if not verb_total:  # For efficiency
            verb_total = self.retrieve_verb_total(verb)
        try:
            return math.log(self.get_conditional_prob(verb, noun, verb_total=verb_total))
        except ValueError:
            return 'None'

    def get_noun_prob(self, noun, noun_totals, noun_sum):
        """Returns P(noun)"""
        return noun_totals[noun] / noun_sum

    def get_sps(self, verb, noun_totals=None, noun_sum=None):
        """Calculates and returns SPS(verb)"""
        try:  # If it never occurs in the corpus, return None
            self.d[verb]
        except KeyError:
            return None
        SPS = 0
        if not noun_sum:
            noun_sum = self.retrieve_noun_sum()
        verb_total = self.retrieve_verb_total(verb)
        if not noun_totals:
            noun_totals = self.retrieve_noun_totals()
        for noun in self.d[verb].keys():
            if noun:
                prior_prob = self.get_noun_prob(noun, noun_totals, noun_sum)
                post_prob = self.get_conditional_prob(verb, noun, verb_total=verb_total)
                SPS += post_prob * math.log(post_prob / prior_prob)
        return SPS

    def get_sa(self, verb, noun, noun_totals=None, noun_sum=None, sps=None):
        """Calculates and returns SA(verb)"""
        try:  # If it never occurs in the corpus, return None
            self.d[verb][noun]
        except KeyError:
            return None
        if not noun_sum:
            noun_sum = self.retrieve_noun_sum()
        if not noun_totals:
            noun_totals = self.retrieve_noun_totals()
        SA = 1 / self.get_sps(verb, noun_totals=noun_totals, noun_sum=noun_sum)
        SA *= self.get_conditional_prob(verb, noun)
        SA *= math.log(self.get_conditional_prob(verb, noun) / self.get_noun_prob(noun, noun_totals, noun_sum))
        return SA

    def print_metrics(self, verb, noun, noun_totals=None, noun_sum=None):
        """Returns conditional probability, SPS and SA"""
        if not noun_sum:
            noun_sum = self.retrieve_noun_sum()
        if not noun_totals:
            noun_totals = self.retrieve_noun_totals()
        print('Number of nouns occurring with for verb %s: %d' % (verb, len(self.d[verb])))
        print('Count of noun %s: %d' % (noun, noun_totals[noun]))
        print('Count of %s and %s: %d' % (verb, noun, self.retrieve_count(verb, noun)))
        print('SPS of %s: %s' % (verb, self.get_sps(verb, noun_totals=noun_totals, noun_sum=noun_sum)))
        print('Conditional probability of %s and %s: %s' % (verb, noun, self.get_conditional_prob(verb, noun)))
        if self.get_conditional_prob(verb, noun) > 0:
            print('Log conditional probability of %s and %s: %s' % (
                verb, noun, math.log(self.get_conditional_prob(verb, noun))))
        if self.get_sa(verb, noun, noun_totals=noun_totals, noun_sum=noun_sum):
            print('SA of %s and %s: %.2fe-03' % (
                verb, noun, 1000 * self.get_sa(verb, noun, noun_totals=noun_totals, noun_sum=noun_sum)))
        print('')

    def get_metrics(self, verb, noun, noun_totals=None, noun_sum=None):
        """Returns conditional probability, SPS and SA"""
        if not noun_sum:
            noun_sum = self.retrieve_noun_sum()
        if not noun_totals:
            noun_totals = self.retrieve_noun_totals()
        cond_prob = self.get_conditional_prob(verb, noun)
        log_prob = self.get_log_prob(verb, noun)
        sps = self.get_sps(verb, noun_totals=noun_totals, noun_sum=noun_sum)
        sa = self.get_sa(verb, noun, noun_totals=noun_totals, noun_sum=noun_sum)
        return cond_prob, log_prob, sps, sa
