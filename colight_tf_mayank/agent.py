"""
Creation Date : 7th Dec 2021
Last Updated : 7th Dec 2021
Author/s : Mayank Sharan

File containing the base Agent class
"""


class Agent(object):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id="0"):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.intersection_id = intersection_id

    def choose_action(self):

        raise NotImplementedError
