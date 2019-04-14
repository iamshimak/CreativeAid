"""Utilities for extracting probabilities and measures from counts dict"""

import logging
import math


class WordFrequency(object):

    def __init__(self, frequency_model):
        self.d = frequency_model

    def count(self, verb, noun):
        """Returns count from dict, given a verb and a noun"""
        try:
            return float(self.d[verb][noun])
        except KeyError:
            return 0

    def verb_total(self, verb):
        """Returns total count for a verb"""
        try:
            return float(sum(self.d[verb].values()))
        except KeyError:
            return 0

    def noun_total(self, noun):
        """Returns total count for a noun"""
        total = 0
        for verb in self.d:
            try:
                total += self.d[verb][noun]
            except KeyError:
                pass
        return float(total)

    def noun_totals(self):
        """Returns dict of all total counts for all nouns"""
        noun_totals = {}
        noun_set = self.noun_set()
        for noun in noun_set:  # Initialize dict
            noun_totals[noun] = 0
        for verb in self.d:
            for noun in self.d[verb]:
                noun_totals[noun] += self.d[verb][noun]
        return noun_totals

    def noun_sum(self):
        """Returns sum of noun counts in the dict"""
        total = 0
        for verb in self.d:
            total += sum(self.d[verb].values())
        return float(total)

    def noun_set(self):
        """Returns set of all nouns in the dict"""
        noun_list = []
        for verb in self.d:
            noun_list += self.d[verb].keys()
        return set(noun_list)

    def conditional_prob(self, verb, noun, verb_total=None):
        """Returns P(noun|verb)"""
        if not verb_total:  # For efficiency
            verb_total = self.verb_total(verb)
        try:
            return self.count(verb, noun) / verb_total
        except ZeroDivisionError:
            return 0

    def log_prob(self, verb, noun, verb_total=None):
        """Returns log(P(noun|verb))"""
        if not verb_total:  # For efficiency
            verb_total = self.verb_total(verb)
        try:
            return math.log(self.conditional_prob(verb, noun, verb_total=verb_total))
        except ValueError:
            return 'None'

    def noun_prob(self, noun, noun_totals, noun_sum):
        """Returns P(noun)"""
        return noun_totals[noun] / noun_sum

    def sps(self, verb, noun_totals=None, noun_sum=None):
        """Calculates and returns SPS(verb)"""
        try:  # If it never occurs in the corpus, return None
            self.d[verb]
        except KeyError:
            return None
        SPS = 0
        if not noun_sum:
            noun_sum = self.noun_sum()
        verb_total = self.verb_total(verb)
        if not noun_totals:
            noun_totals = self.noun_totals()
        for noun in self.d[verb].keys():
            if noun:
                prior_prob = self.noun_prob(noun, noun_totals, noun_sum)
                post_prob = self.conditional_prob(verb, noun, verb_total=verb_total)
                SPS += post_prob * math.log(post_prob / prior_prob)
        return SPS

    def sa(self, verb, noun, noun_totals=None, noun_sum=None, sps=None):
        """Calculates and returns SA(verb)"""
        try:  # If it never occurs in the corpus, return None
            self.d[verb][noun]
        except KeyError:
            return None
        if not noun_sum:
            noun_sum = self.noun_sum()
        if not noun_totals:
            noun_totals = self.noun_totals()
        SA = 1 / self.sps(verb, noun_totals=noun_totals, noun_sum=noun_sum)
        SA *= self.conditional_prob(verb, noun)
        SA *= math.log(self.conditional_prob(verb, noun) / self.noun_prob(noun, noun_totals, noun_sum))
        return SA

    def print_metrics(self, verb, noun, noun_totals=None, noun_sum=None):
        """Returns conditional probability, SPS and SA"""
        if not noun_sum:
            noun_sum = self.noun_sum()
        if not noun_totals:
            noun_totals = self.noun_totals()
        print('Number of nouns occurring with for verb %s: %d' % (verb, len(self.d[verb])))
        print('Count of noun %s: %d' % (noun, noun_totals[noun]))
        print('Count of %s and %s: %d' % (verb, noun, self.count(verb, noun)))
        print('SPS of %s: %s' % (verb, self.sps(verb, noun_totals=noun_totals, noun_sum=noun_sum)))
        print('Conditional probability of %s and %s: %s' % (verb, noun, self.conditional_prob(verb, noun)))
        if self.conditional_prob(verb, noun) > 0:
            print('Log conditional probability of %s and %s: %s' % (
                verb, noun, math.log(self.conditional_prob(verb, noun))))
        if self.sa(verb, noun, noun_totals=noun_totals, noun_sum=noun_sum):
            print('SA of %s and %s: %.2fe-03' % (
                verb, noun, 1000 * self.sa(verb, noun, noun_totals=noun_totals, noun_sum=noun_sum)))
        print('')

    def metrics(self, verb, noun, noun_totals=None, noun_sum=None):
        """Returns conditional probability, SPS and SA"""
        if not noun_sum:
            noun_sum = self.noun_sum()
        if not noun_totals:
            noun_totals = self.noun_totals()
        cond_prob = self.conditional_prob(verb, noun)
        log_prob = self.log_prob(verb, noun)
        sps = self.sps(verb, noun_totals=noun_totals, noun_sum=noun_sum)
        sa = self.sa(verb, noun, noun_totals=noun_totals, noun_sum=noun_sum)
        return cond_prob, log_prob, sps, sa
