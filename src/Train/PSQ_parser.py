import numpy as np
import os
import glob

class PSQParser:
    def __init__(self, board_size=20):
        self.board_size = board_size

    def augment_board(self, board_2d, policy_2d, last_move_channel):
        """
        Returns a list of 8 variations (rotations/flips) of the board and policy.
        """
        augmented_X = []
        augmented_Y = []
        augmented_last_move = []

        for k in range(1): # 0, 90, 180, 270 degrees
            rot_X = np.rot90(board_2d, k)
            rot_Y = np.rot90(policy_2d, k)
            rot_last_move = np.rot90(last_move_channel, k)

            augmented_X.append(rot_X)
            augmented_Y.append(rot_Y)
            augmented_last_move.append(rot_last_move)

            # Mirror (Flip Left-Right)
            augmented_X.append(np.fliplr(rot_X))
            augmented_Y.append(np.fliplr(rot_Y))
            augmented_last_move.append(np.fliplr(rot_last_move))

        return augmented_X, augmented_Y, augmented_last_move

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
                    winner = 1 # Black Wins
                elif res_code == 2:
                    winner = -1 # White Wins
                elif res_code == 0:
                    winner = 0 # Draw
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

                # Sanity check
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

                aug_X, aug_Y, aug_last_move = self.augment_board(board_2d, policy_2d, last_move_channel)
                for ax, ay, al in zip(aug_X, aug_Y, aug_last_move):
                    chan_my = (ax == 1).astype(np.float32)
                    chan_op = (ax == -1).astype(np.float32)
                    chan_last = al.astype(np.float32)

                    merge_input = np.concatenate([chan_my.flatten(), chan_op.flatten(), chan_last.flatten()], axis=0)

                    X_list.append(merge_input)
                    Y_policy_list.append(ay.flatten())
                    outcome = 1.0 if current_player == winner else -1.0
                    Y_value_list.append(np.array([outcome], dtype=np.float32))

                prev_x, prev_y = x, y

                # Apply move to board
                board[y, x] = current_player

                # Switch turn
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