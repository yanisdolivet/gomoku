import numpy as np
import os
import glob
from scipy.signal import convolve2d

class PSQParser:
    def __init__(self, board_size=20):
        self.board_size = board_size

    def get_threat_mask(self, board, player):
        """
        Generates a 20x20 mask where 1.0 means 'Playing here creates a 4 or 5'.
        Mimics C++ getWinningCandidates and getThreatCandidates.
        """
        player_stones = (board == player).astype(int)
        
        kernels = [
            np.array([[1, 1, 1, 1]]),             # Horizontal (radius 4)
            np.array([[1], [1], [1], [1]]),       # Vertical
            np.eye(4, dtype=int),                 # Diagonal \
            np.fliplr(np.eye(4, dtype=int))       # Diagonal /
        ]
        
        threat_mask = np.zeros_like(board, dtype=np.float32)
        
        for k in kernels:
            count = convolve2d(player_stones, k, mode='same')
            pass
        
        rows, cols = board.shape
        total_threats = np.zeros((rows, cols), dtype=bool)
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dy, dx in directions:
            p = (board == player)
            
            def shift(arr, steps_y, steps_x):
                res = np.roll(arr, (steps_y, steps_x), axis=(0, 1))
                if steps_y > 0: res[:steps_y, :] = False
                elif steps_y < 0: res[steps_y:, :] = False
                if steps_x > 0: res[:, :steps_x] = False
                elif steps_x < 0: res[:, steps_x:] = False
                return res

            f1 = shift(p, dy*1, dx*1)
            f2 = shift(p, dy*2, dx*2)
            f3 = shift(p, dy*3, dx*3)
            
            b1 = shift(p, -dy*1, -dx*1)
            b2 = shift(p, -dy*2, -dx*2)
            b3 = shift(p, -dy*3, -dx*3)
           
            c1 = b1 & b2 & b3 
            c2 = f1 & f2 & f3
            c3 = b1 & f1 & f2
            c4 = b1 & b2 & f1
            
            total_threats |= (c1 | c2 | c3 | c4)

        valid_threats = total_threats & (board == 0)
        
        return valid_threats.astype(np.float32)

    def augment_board(self, board_2d, policy_2d, last_move_channel, my_threat_mask, op_threat_mask):
        """
        Returns a list of 8 variations (rotations/flips) of the board and policy.
        """
        augmented_X = []
        augmented_Y = []
        augmented_last_move = []
        augmented_my_threat_mask = []
        augmented_op_threat_mask = []

        for k in range(4):
            rot_X = np.rot90(board_2d, k)
            rot_Y = np.rot90(policy_2d, k)
            rot_last_move = np.rot90(last_move_channel, k)
            rot_my_threat_mask = np.rot90(my_threat_mask, k)
            rot_op_threat_mask = np.rot90(op_threat_mask, k)

            augmented_X.append(rot_X)
            augmented_Y.append(rot_Y)
            augmented_last_move.append(rot_last_move)
            augmented_my_threat_mask.append(rot_my_threat_mask)
            augmented_op_threat_mask.append(rot_op_threat_mask)

            augmented_X.append(np.fliplr(rot_X))
            augmented_Y.append(np.fliplr(rot_Y))
            augmented_last_move.append(np.fliplr(rot_last_move))
            augmented_my_threat_mask.append(np.fliplr(rot_my_threat_mask))
            augmented_op_threat_mask.append(np.fliplr(rot_op_threat_mask))

        return augmented_X, augmented_Y, augmented_last_move, augmented_my_threat_mask, augmented_op_threat_mask

    def parse_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
        except Exception:
            return [], [], []

        if not lines:
            return [], [], []

        header = lines[0]
        if "20x20" not in header and str(self.board_size) not in header:
            return [], [], []

        winner = 0

        result_line = lines[-1]
        if ',' in result_line:
            parts = result_line.split(',')
            try:
                res_code = int(parts[0])
                if res_code == 1:
                    winner = 1
                elif res_code == 2:
                    winner = -1
                elif res_code == 0:
                    winner = 0
            except ValueError:
                pass

        if winner == 0:
            return [], [], []

        X_list = []
        Y_policy_list = []
        Y_value_list = []

        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        current_player = 1
        prev_x, prev_y = -1, -1

        for line in lines[1:]:
            if line == '-1' or 'zip' in line or 'Freestyle' in line:
                break

            try:
                parts = line.split(',')
                if len(parts) < 2: continue

                x = int(parts[0]) - 1
                y = int(parts[1]) - 1

                if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                    continue

                input_state = board.copy()
                if current_player == -1:
                    input_state = -input_state

                last_move_channel = np.zeros((self.board_size, self.board_size), dtype=np.float32)
                if prev_x != -1 and prev_y != -1:
                    last_move_channel[prev_y, prev_x] = 1.0

                board_2d = input_state.reshape(self.board_size, self.board_size)
                policy_2d = np.zeros((self.board_size, self.board_size), dtype=np.float32)
                policy_2d[y, x] = 1.0

                my_threat_mask = self.get_threat_mask(board_2d, current_player)
                op_threat_mask = self.get_threat_mask(board_2d, -current_player)

                aug_X, aug_Y, aug_last_move, aug_my_threat_mask, aug_op_threat_mask = self.augment_board(board_2d, policy_2d, last_move_channel, my_threat_mask, op_threat_mask)
                for ax, ay, al, am, ao in zip(aug_X, aug_Y, aug_last_move, aug_my_threat_mask, aug_op_threat_mask):
                    chan_my = (ax == 1).astype(np.float32)
                    chan_op = (ax == -1).astype(np.float32)
                    chan_last = al.astype(np.float32)
                    chan_my_threat = am.astype(np.float32)
                    chan_op_threat = ao.astype(np.float32)

                    merge_input = np.concatenate([chan_my.flatten(), chan_op.flatten(), chan_last.flatten(), chan_my_threat.flatten(), chan_op_threat.flatten() ], axis=0)

                    X_list.append(merge_input)
                    Y_policy_list.append(ay.flatten())
                    outcome = 1.0 if current_player == winner else -1.0
                    Y_value_list.append(np.array([outcome], dtype=np.float32))

                prev_x, prev_y = x, y

                board[y, x] = current_player
                current_player = -current_player

            except ValueError:
                continue

        return X_list, Y_policy_list, Y_value_list

    def load_dataset(self, folders_path):
        all_X, all_Yp, all_Yv = [], [], []
        files = []

        for folder in folders_path:
            file_in_dir = glob.glob(os.path.join(folder, "*.psq"))
            if not file_in_dir:
                file_in_dir = glob.glob(os.path.join(folder, "*.txt"))
            files.extend(file_in_dir)

        print(f"Found {len(files)} PSQ files.")

        for i, f in enumerate(files):
            x, yp, yv = self.parse_file(f)
            all_X.extend(x)
            all_Yp.extend(yp)
            all_Yv.extend(yv)
            if (i+1) % 100 == 0: print(f"Parsed {i+1} files...")

        return np.array(all_X), np.array(all_Yp), np.array(all_Yv)