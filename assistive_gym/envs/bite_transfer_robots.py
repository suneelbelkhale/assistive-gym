from .bite_transfer import BiteTransferEnv
#
# class FeedingPR2Env(FeedingEnv):
#     def __init__(self):
#         super(FeedingPR2Env, self).__init__(robot_type='pr2', human_control=False)
#
# class FeedingBaxterEnv(FeedingEnv):
#     def __init__(self):
#         super(FeedingBaxterEnv, self).__init__(robot_type='baxter', human_control=False)
#
# class FeedingSawyerEnv(FeedingEnv):
#     def __init__(self):
#         super(FeedingSawyerEnv, self).__init__(robot_type='sawyer', human_control=False)
#
# class FeedingJacoEnv(FeedingEnv):
#     def __init__(self):
#         super(FeedingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class BiteTransferPandaEnv(BiteTransferEnv):
    def __init__(self):
        super(BiteTransferPandaEnv, self).__init__(robot_type='panda', human_control=False)
#
# class FeedingPR2HumanEnv(FeedingEnv):
#     def __init__(self):
#         super(FeedingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)
#
# class FeedingBaxterHumanEnv(FeedingEnv):
#     def __init__(self):
#         super(FeedingBaxterHumanEnv, self).__init__(robot_type='baxter', human_control=True)
#
# class FeedingSawyerHumanEnv(FeedingEnv):
#     def __init__(self):
#         super(FeedingSawyerHumanEnv, self).__init__(robot_type='sawyer', human_control=True)
#
# class FeedingJacoHumanEnv(FeedingEnv):
#     def __init__(self):
#         super(FeedingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)
#
class BiteTransferPandaHumanEnv(BiteTransferEnv):
    def __init__(self):
        super(BiteTransferPandaHumanEnv, self).__init__(robot_type='panda', human_control=True)

