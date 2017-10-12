import sys
import numpy as np
from matplotlib import pyplot as plt

class LaneProba():
    """
    Stores the foreground, background and width probabilities for a Lane
    """
    def __init__(self, proba=0.1):
        self.set_proba(proba)

    def decrease(self, ratio=0.8):
        self.foreground_position_proba *= ratio
        self.background_position_proba *= ratio
        self.width_proba *= ratio

    def increase(self, ratio=1.2):
        self.foreground_position_proba *= ratio
        self.background_position_proba *= ratio
        self.width_proba *= ratio

        # cap proba to 1.0 max
        self.foreground_position_proba = min(1.0, self.foreground_position_proba)
        self.background_position_proba = min(1.0, self.background_position_proba)
        self.width_proba = min(1.0, self.width_proba)

    def set_proba(self, proba):
        self.foreground_position_proba = proba
        self.background_position_proba = proba
        self.width_proba = proba

    def get_values(self):
        return np.array([self.foreground_position_proba, self.background_position_proba, self.width_proba])

    def __str__(self):
        return "probas:{} {} {}".format(self.foreground_position_proba,
                                        self.background_position_proba,
                                        self.width_proba)


class Lane():
    def __init__(self, lane_left, lane_right):
        # lane: [nb_points, yx]
        foreground_left  = lane_left[-1]
        foreground_right = lane_right[-1]

        background_left  = lane_left[0]
        background_right  = lane_right[0]

        self.foreground_position = (foreground_left + foreground_right) / 2
        self.background_position = (background_left + background_right) / 2
        self.width = foreground_right - foreground_left

        self.skeleton = lane_left+ (lane_right - lane_left)/2.0
        self.probas = None

        # Check validity of the lane
        background_width_invalid = (background_right - background_left) > 2*self.width
        line_crossing = np.any(lane_right < lane_left)
        width_too_small = np.mean(lane_right - lane_left) < 5

        self.is_invalid = background_width_invalid or line_crossing or width_too_small
        #if self.is_invalid:
        #    print("lane is invalid !")

    def get_parameters(self):
        return np.array([self.foreground_position, self.background_position, self.width])

    def set_skeleton(self, sk):
        self.skeleton = sk
        self.foreground_position = int(sk[-1])
        self.background_position = int(sk[0])

    def set_width(self, w):
        self.width = w

    def set_probas(self, lane_probas):
        self.probas = lane_probas

    def distance(self, lane):
        foreground_distance = np.abs(self.foreground_position - lane.foreground_position)
        mean_distance = np.abs(np.mean(self.skeleton - lane.skeleton))
        weight_foreground = 0.75
        return foreground_distance * weight_foreground + mean_distance * (1-weight_foreground)

    def __eq__(self, other):
        return np.array_equal(self.skeleton, other.skeleton)

    def __str__(self):
        return "pos:{} back_pos:{} width:{}".format(self.foreground_position, self.background_position,
                                                self.width)

class RoadState():
    def __init__(self, lanes):
        self.current_lanes = []
        for lane in lanes:
            if not lane.is_invalid:
                self.current_lanes.append(lane)

        # Add default probas
        for lane in self.current_lanes:
            lane.set_probas(LaneProba(0.8))

    def get_probas(self):
        nb_lanes = len(self.current_lanes)
        probas = np.zeros((nb_lanes, 3))
        for i in range(nb_lanes):
            lane = self.current_lanes[i]
            if lane is None:
                probas[i] = 0
            else:
                probas[i] = lane.probas.get_values()
        return probas

    def compute_proba(self, value, comparison):
        if comparison == 0:
            return 0

        ratio = abs(value / comparison)
        if ratio < 0.1:
            proba = 1.0
        elif ratio < 0.2:
            proba = 0.9
        elif ratio < 0.5:
            proba = 0.8
        elif ratio < 1.0:
            proba = 0.5
        else:
            proba = 0.3

        return proba

    def compute_probas(self, parameters, comparison):
        probas = [self.compute_proba(value, comparison) for value in parameters]
        return np.array(probas)

    def compute_changes(self, new_state):
        nb_lanes = len(self.current_lanes)
        all_diffs_parameters = np.zeros((nb_lanes, 3)) # fore, back, width
        new_probas = np.zeros((nb_lanes, 3))

        for i in range(nb_lanes):
            old_lane = self.current_lanes[i]
            new_lane = new_state.current_lanes[i]

            if new_lane is None:
                new_state.current_lanes[i] = old_lane
                all_diffs_parameters[i] = 0
                new_probas[i] = 0.2
            else:
                diffs_parameters = new_lane.get_parameters() - old_lane.get_parameters()
                diffs_skeleton = new_lane.skeleton - old_lane.skeleton

                distance_parameters = np.abs(diffs_parameters)
                distance_skeletons = np.abs(diffs_skeleton)

                probas_parameters = self.compute_probas(distance_parameters, old_lane.width)

                all_diffs_parameters[i] = diffs_parameters
                new_probas[i] = probas_parameters

        return all_diffs_parameters, new_probas

    def __str__(self):
        state_str_list = []
        for lane in self.current_lanes:
            if lane is None:
                state_str_list.append("None")
            else:
                state_str_list.append(str(lane.foreground_position) + '/' + str(lane.background_position))
        return " ".join(state_str_list)

class LanesManager():

    def update_lines(self, lines_list):
        raise NotImplementedError

class SimpleLanesManager(LanesManager):
    def __init__(self, record=False):
        self.record = record
        self.current_road_state = None

    def update_lines(self, lines_list):
        # print("")
        if self.record:
            f_handle = file("output.txt", 'a')
            np.save(f_handle, lines_list)
            f_handle.close()

        # New lanes
        new_lanes = []
        for left, right in list(zip(lines_list, lines_list[1:])):
            new_lanes.append(Lane(left, right))

        if self.current_road_state == None:
            # init
            self.current_road_state = RoadState(new_lanes)
        else:
            # New road state
            new_road_state = RoadState(new_lanes)

            '''
            print()
            print("Begin:")
            print("current state: ", str(self.current_road_state))
            print("new state:     ", str(new_road_state))
            print("current probas:", self.current_road_state.get_probas())
            print("new probas:", new_road_state.get_probas())
            '''

            SimpleLanesManager.prepare_for_merge(self.current_road_state, new_road_state)

            SimpleLanesManager.fix_missing_lanes(self.current_road_state, new_road_state)

            '''
            print("Before merge:")
            print("current state: ", str(self.current_road_state))
            print("new state:     ", str(new_road_state))
            print("current probas:", self.current_road_state.get_probas())
            print("new probas:", new_road_state.get_probas())
            '''

            SimpleLanesManager.merge(self.current_road_state, new_road_state)

            '''
            print("End:")
            print("final state: ", str(self.current_road_state))
            print("final probas:", self.current_road_state.get_probas())
            '''


    @staticmethod
    def prepare_for_merge(current_road_state, new_road_state):
        """
        Prepare both models before merge operation:
            - Pair old lanes with new lanes
            - Insert Fake lanes in both model when the number of lanes do not match, in order to compare the lanes
              easily in the next steps
        :param current_road_state: current model
        :param new_road_state: new candidate model
        :return: nothing, both models are updated directly and should now have the same number of lanes. Missing lanes
        in each model are tagged 'None'
        """

        new_lanes = new_road_state.current_lanes

        # 1. Find a mapping between the lanes of the old model and the new lanes

        # Get distance table: distance between old and new lanes
        max_number_of_lanes = max(len(current_road_state.current_lanes), len(new_lanes))
        distance_table = np.ones([max_number_of_lanes, max_number_of_lanes]) * 200 # 200 == infinity
        for i in range(len(current_road_state.current_lanes)):
            for j in range(len(new_lanes)):
                distance_table[i, j] = new_lanes[j].distance(current_road_state.current_lanes[i])

        #print('Distance_table:\n{}\n'.format(distance_table.astype(np.int)))

        # reorder distance table in f() of min distance for each new lane
        new_lanes_minimum_distances = np.min(distance_table, axis=0)
        index_new_lanes_discover = np.argsort(new_lanes_minimum_distances)

        # compute the mapping old lane <-> new lane
        new_lanes_2_current_lanes_mapping = np.ones(max_number_of_lanes, dtype=int) * -1
        for j in index_new_lanes_discover:
            index_distance_min = np.argmin(distance_table[:, j])
            new_lanes_2_current_lanes_mapping[j] = index_distance_min
            distance_table[index_distance_min, :] = 1000

        # 2. Fix consistency issues between old/new model: each model must have the same number of lanes so that
        # it is easy to compare them. This step adds 'None' elements before, after or between lanes inside the
        # old model and new model when needed (i.e. missing lanes are now tagged as 'None').

        for j in index_new_lanes_discover:
            corresponding_current_lane = new_lanes_2_current_lanes_mapping[j]

            if corresponding_current_lane >= len(current_road_state.current_lanes):
                # New lane
                inserted = False
                for i in range(len(current_road_state.current_lanes)):
                    if new_lanes[j].foreground_position < current_road_state.current_lanes[i].foreground_position:
                        current_road_state.current_lanes.insert(i, None)
                        inserted = True
                        break
                # could not find a place on the left or middle, add on the right
                if not inserted:
                    current_road_state.current_lanes.append(None)

            elif j >= len(new_lanes):
                # lane disappeared

                # lane on the left side disappeared
                inserted = False
                current_lane = current_road_state.current_lanes[corresponding_current_lane]
                for i in range(len(new_road_state.current_lanes)):
                    new_lane = new_road_state.current_lanes[i]
                    if new_lane is None:
                        continue
                    if current_lane.foreground_position < new_lane.foreground_position:
                        new_road_state.current_lanes.insert(i, None)
                        inserted = True
                        break
                # could not find a place on the left or middle, add on the right
                if not inserted:
                    new_road_state.current_lanes.append(None)

            elif corresponding_current_lane == -1:
                # -1 : new lane didn't exist previously
                raise Exception("Should not happen")
            else:
                # Nothing to do
                pass

    # add missing lanes in old/new model
    @staticmethod
    def fix_missing_lanes(current_road_state, new_road_state):
        """
        Update the missing lanes tagged as 'None' by previous step.
           - When a new lane appears in the new model, add it to the old model as well (so that we can compare models)
           - When a lane disappears in the new model, add the old lane in the new model but with a lower probability
        :param current_road_state: current model
        :param new_road_state: new candidate model
        :return: nothing: both models are updated directly
        """

        assert len(current_road_state.current_lanes) == len(new_road_state.current_lanes)

        for i in range(len(current_road_state.current_lanes)):
            lane_before, lane_after = current_road_state.current_lanes[i], new_road_state.current_lanes[i]
            assert not ((lane_before is None) and (lane_after is None)) # one lane must exist

            if lane_before is None:
                # New lane appeared.
                # Add it in the old model and set its proba to 0.1
                lane_after.set_probas(LaneProba())
                current_road_state.current_lanes[i] = lane_after

            elif lane_after is None:
                # Lane disappeared. Decrease proba and use for new model
                lane_before.probas.decrease()
                #new_road_state.current_lanes[i] = lane_before

    @staticmethod
    def merge(current_road_state, new_road_state):
        """
        Merge 2 the new road model into the current road model.
        :param current_road_state: current model
        :param new_road_state: new candidate model
        :return: merged model
        """

        old_probas = current_road_state.get_probas()

        # Compute the differences between old road state and the new one,
        # and the probabilities for each difference
        diffs, new_probas = current_road_state.compute_changes(new_road_state)

        # compute best diff
        #print("old probas:", old_probas)
        #print("new probas:", new_probas)

        skeletons_before = np.array([lane.skeleton for lane in current_road_state.current_lanes])
        skeletons_after = np.array([lane.skeleton for lane in new_road_state.current_lanes])

        skeletons_diffs = skeletons_after - skeletons_before
        mean_skeleton_diffs = np.median(skeletons_diffs, axis=0)

        mean_skeleton_after = skeletons_before + mean_skeleton_diffs

        for i in range(len(current_road_state.current_lanes)):
            lane_before, lane_after = current_road_state.current_lanes[i], new_road_state.current_lanes[i]
            sk1 = lane_before.skeleton

            p1 = np.mean(old_probas[i])
            p2 = np.mean(new_probas[i])
            current_value_proba = p1 / (p1 + p2)

            if p1 <= 0.25:
                # probability for this skeleton is too low : do not use it, but instead compute
                # the mean change from the other skeletons in the road and apply this orientation to the
                # current skeleton
                current_sk = mean_skeleton_after[i]
                other_sks = mean_skeleton_after[np.arange(len(mean_skeleton_after)) != i]
                dif = current_sk - other_sks
                dif = dif - dif[:, -1].reshape(-1, 1)
                mean_dif = np.mean(dif, axis=0)
                sk2 = lane_before.skeleton - mean_dif

                # debug:
                #sk2 = mean_skeleton_after[i]
            else:
                sk2 = lane_after.skeleton

            # Compute new skeleton,  between old/new skeleton weighted by the current proba
            new_sk = sk1 * current_value_proba + sk2 * (1.0 - current_value_proba)
            current_road_state.current_lanes[i].set_skeleton(new_sk)

            # Same for the width
            new_width = lane_before.width * current_value_proba + lane_after.width * (1.0 - current_value_proba)
            current_road_state.current_lanes[i].set_width(new_width)

            # Update the proba of the lane
            debug = False
            if not debug:
                current_road_state.current_lanes[i].probas.set_proba(p2)
            else:
                # manually increase/decrease probas step by step
                if p1 > p2:
                    # previous probability was higher than the new one : decrease the proba for this lane
                    current_road_state.current_lanes[i].probas.decrease()
                elif p2 > p1:
                    # new proba is better : increase proba for this lane
                    current_road_state.current_lanes[i].probas.increase()
