#!/usr/bin/env python3
"""
Board Log Parser
Parses and displays Gomoku game logs in a clean, readable format.
"""

import re
import sys


def strip_ansi_codes(text):
    """Remove ANSI color codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def parse_board_log(filepath):
    """Parse the board log file and extract game information."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by moves
    moves = re.split(r'Move \d+:', content)[1:]  # Skip empty first element
    
    game_data = []
    for move_text in moves:
        lines = move_text.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Parse move info
        player_line = lines[0].strip()
        time_line = lines[1].strip()
        
        player_match = re.match(r'Player(\d+):\s*(\d+),(\d+)', player_line)
        time_match = re.match(r'Time:\s*([\d.]+)ms', time_line)
        
        if player_match and time_match:
            player = int(player_match.group(1))
            row = int(player_match.group(2))
            col = int(player_match.group(3))
            time_ms = float(time_match.group(1))
            
            # Extract board state (clean version)
            board_lines = lines[2:]  # Skip separator line if present
            if board_lines and '----' in board_lines[0]:
                board_lines = board_lines[1:]
            
            # Clean board
            clean_board = []
            for line in board_lines:
                if line.strip():  # Skip empty lines
                    clean_line = strip_ansi_codes(line)
                    clean_board.append(clean_line)
            
            game_data.append({
                'move_num': len(game_data) + 1,
                'player': player,
                'row': row,
                'col': col,
                'time_ms': time_ms,
                'board': clean_board
            })
    
    return game_data


def display_move_list(game_data):
    """Display a list of all moves."""
    print("=" * 80)
    print("GAME SUMMARY")
    print("=" * 80)
    print(f"{'Move':<6} {'Player':<8} {'Position':<12} {'Time':<15} {'Notes':<20}")
    print("-" * 80)
    
    for move in game_data:
        player_symbol = 'O' if move['player'] == 1 else 'X'
        position = f"({move['row']}, {move['col']})"
        time_str = f"{move['time_ms']:.3f} ms"
        
        # Classify move speed
        if move['time_ms'] < 1:
            note = "Instant (forced/book)"
        elif move['time_ms'] < 100:
            note = "Very fast"
        elif move['time_ms'] < 1000:
            note = "Fast"
        else:
            note = "Thinking..."
        
        print(f"{move['move_num']:<6} Player{move['player']} ({player_symbol}){'':<2} {position:<12} {time_str:<15} {note:<20}")
    
    print("-" * 80)
    print(f"Total moves: {len(game_data)}")
    
    # Calculate average times
    p1_times = [m['time_ms'] for m in game_data if m['player'] == 1]
    p2_times = [m['time_ms'] for m in game_data if m['player'] == 2]
    
    if p1_times:
        print(f"Player 1 (O) average time: {sum(p1_times)/len(p1_times):.2f} ms")
    if p2_times:
        print(f"Player 2 (X) average time: {sum(p2_times)/len(p2_times):.2f} ms")
    print()


def display_board_at_move(game_data, move_num):
    """Display the board state at a specific move."""
    if move_num < 1 or move_num > len(game_data):
        print(f"Invalid move number. Must be between 1 and {len(game_data)}")
        return
    
    move = game_data[move_num - 1]
    player_symbol = 'O' if move['player'] == 1 else 'X'
    
    print("=" * 80)
    print(f"MOVE {move['move_num']}: Player {move['player']} ({player_symbol}) plays at ({move['row']}, {move['col']})")
    print(f"Time: {move['time_ms']:.3f} ms")
    print("=" * 80)
    
    # Add column numbers
    print("    ", end="")
    for col in range(20):
        print(f"{col:2}", end="")
    print()
    
    # Display board with row numbers
    for row_num, line in enumerate(move['board'][:20]):  # Limit to 20 rows
        print(f"{row_num:2}: {line}")
    print()


def display_final_board(game_data):
    """Display the final board state."""
    if not game_data:
        print("No game data available.")
        return
    
    display_board_at_move(game_data, len(game_data))


def interactive_mode(game_data):
    """Interactive mode to browse through moves."""
    current_move = len(game_data)  # Start at final position
    
    while True:
        display_board_at_move(game_data, current_move)
        
        print("Commands:")
        print("  n/next     - Next move")
        print("  p/prev     - Previous move")
        print("  f/first    - First move")
        print("  l/last     - Last move")
        print("  g <num>    - Go to move number")
        print("  s/summary  - Show move summary")
        print("  q/quit     - Quit")
        print()
        
        command = input("Enter command: ").strip().lower()
        
        if command in ['q', 'quit']:
            break
        elif command in ['n', 'next']:
            current_move = min(current_move + 1, len(game_data))
        elif command in ['p', 'prev']:
            current_move = max(current_move - 1, 1)
        elif command in ['f', 'first']:
            current_move = 1
        elif command in ['l', 'last']:
            current_move = len(game_data)
        elif command in ['s', 'summary']:
            display_move_list(game_data)
            input("\nPress Enter to continue...")
        elif command.startswith('g '):
            try:
                move_num = int(command.split()[1])
                if 1 <= move_num <= len(game_data):
                    current_move = move_num
                else:
                    print(f"Invalid move number. Must be between 1 and {len(game_data)}")
                    input("Press Enter to continue...")
            except (ValueError, IndexError):
                print("Invalid command. Use 'g <number>'")
                input("Press Enter to continue...")
        else:
            print("Unknown command.")
            input("Press Enter to continue...")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: ./parse_board_log.py <board_log_file> [options]")
        print("\nOptions:")
        print("  --summary          Show move summary")
        print("  --final            Show final board position")
        print("  --move <num>       Show board at specific move")
        print("  --interactive      Interactive mode (default)")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        game_data = parse_board_log(filepath)
        
        if len(game_data) == 0:
            print("No game data found in the log file.")
            sys.exit(1)
        
        # Handle command-line options
        if len(sys.argv) > 2:
            option = sys.argv[2]
            
            if option == '--summary':
                display_move_list(game_data)
            elif option == '--final':
                display_final_board(game_data)
            elif option == '--move' and len(sys.argv) > 3:
                try:
                    move_num = int(sys.argv[3])
                    display_board_at_move(game_data, move_num)
                except ValueError:
                    print("Invalid move number.")
                    sys.exit(1)
            else:
                print("Unknown option.")
                sys.exit(1)
        else:
            # Default: interactive mode
            interactive_mode(game_data)
            
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
