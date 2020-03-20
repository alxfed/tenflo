# -*- coding: utf-8 -*-
"""...
"""
from durable.lang import *


def main():
    with ruleset('test'):
        # antecedent
        @when_all(m.subject == 'World')
        def say_hello(c):
            # consequent
            print ('Hello {0}'.format(c.m.subject))

    post('test', { 'subject': 'World' })
    return


if __name__ == '__main__':
    main()
    print('\ndone')