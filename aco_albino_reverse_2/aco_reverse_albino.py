import chess
import chess.engine
import numpy as np
import random
import csv
import time

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
pheromone_matrix = np.full((64, 12), 0.1)

stats = {
    "total": 0, "geo_ok": 0, "valid_board": 0,
    "no_premate": 0, "pawn_moves_ok": 0,
    "sf_rejected": 0, "local_search_improved": 0,
    "reverse_1": 0, "reverse_2": 0, "reverse_3": 0, "reverse_4": 0
}

def format_duration(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def count_pieces(board, color, piece_type):
    return sum(
        1 for sq in chess.SQUARES
        if board.piece_at(sq)
        and board.piece_at(sq).color == color
        and board.piece_at(sq).piece_type == piece_type
    )

def choose_sq(options, piece_symbol):
    if not options: return None
    idx = PIECE_TO_INDEX[piece_symbol]
    values = np.array([pheromone_matrix[sq][idx] for sq in options])
    total = values.sum()
    if total == 0: return random.choice(options)
    probs = values / total
    return np.random.choice(options, p=probs)

def find_wK_safe(board):
    bk_sq = board.king(chess.BLACK)
    candidates = []
    for sq in chess.SQUARES:
        if board.piece_at(sq): continue
        if bk_sq and chess.square_distance(sq, bk_sq) <= 1: continue
        board.set_piece_at(sq, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.WHITE
        if not board.is_check():
            candidates.append(sq)
        board.remove_piece_at(sq)
    return random.choice(candidates) if candidates else None

def board_valid(board):
    try:
        if not board.is_valid(): return False
        board2 = board.copy()
        board2.turn = chess.BLACK
        for move in board2.legal_moves:
            tb = board2.copy(); tb.push(move)
            if tb.is_checkmate(): return False
        return True
    except: return False

def white_has_any_mate_in_1(board):
    b = board.copy(); b.turn = chess.WHITE
    for move in b.legal_moves:
        tb = b.copy(); tb.push(move)
        if tb.is_checkmate(): return True
    return False

def verify_with_stockfish(engine, board, pawn_move):
    """
    Ελέγχει ότι η κίνηση πιονιού εμφανίζεται ως #+2 στα top 5.
    ΔΕΝ ελέγχουμε dual — επιτρέπουμε άλλες κινήσεις να είναι #+2.
    """
    try:
        result = engine.analyse(
            board,
            chess.engine.Limit(depth=22),
            multipv=5
        )
        if not isinstance(result, list):
            result = [result]

        for r in result:
            sc = r["score"].white()
            mv = r.get("pv", [None])[0]
            if mv == pawn_move and sc.is_mate() and sc.mate() == 2:
                return True

        return False
    except:
        return False

def is_valid_pawn_key(board, pawn_move):
    if not board.is_legal(pawn_move): return False
    tb = board.copy(); tb.push(pawn_move)
    if tb.is_checkmate(): return False
    black_moves = list(tb.legal_moves)
    if len(black_moves) == 0: return False
    for br in black_moves:
        tb2 = tb.copy(); tb2.push(br)
        if not white_has_any_mate_in_1(tb2):
            return False
    return True

def build_variations_string(board, wP_sq, reverse_moves, pawn_moves_dict):
    """
    Format UCI: pawn_uci:black_uci:mate_uci | ...
    Μία αντιπροσωπευτική variation ανά key move
    """
    parts = []
    for mt in sorted(reverse_moves):
        pm = pawn_moves_dict[mt]
        bc = board.copy(); bc.turn = chess.WHITE
        if not bc.is_legal(pm): continue
        bc.push(pm)
        for bm in bc.legal_moves:
            tb = bc.copy(); tb.push(bm)
            b2 = tb.copy(); b2.turn = chess.WHITE
            for wm in b2.legal_moves:
                t2 = b2.copy(); t2.push(wm)
                if t2.is_checkmate():
                    parts.append(f"{pm.uci()}:{bm.uci()}:{wm.uci()}")
                    break
            else:
                continue
            break
    return " | ".join(parts)

class ReverseAlbinoACO:
    def __init__(self, engine):
        self.engine = engine

    def get_pawn_moves(self, wP_sq):
        wP_file = chess.square_file(wP_sq)
        wP_rank = chess.square_rank(wP_sq)
        return {
            1: chess.Move(wP_sq, chess.square(wP_file,     wP_rank + 1)),
            2: chess.Move(wP_sq, chess.square(wP_file,     wP_rank + 2)),
            3: chess.Move(wP_sq, chess.square(wP_file - 1, wP_rank + 1)),
            4: chess.Move(wP_sq, chess.square(wP_file + 1, wP_rank + 1)),
        }

    def evaluate_and_score(self, board, wP_sq):
        global stats
        if board is None or wP_sq is None: return 0.0, set()

        bc = board.copy(); bc.turn = chess.WHITE
        if white_has_any_mate_in_1(bc): return 0.0, set()

        pawn_moves = self.get_pawn_moves(wP_sq)
        reverse_found = set()

        for move_type, pawn_move in pawn_moves.items():
            if not is_valid_pawn_key(bc, pawn_move):
                continue
            if not verify_with_stockfish(self.engine, bc, pawn_move):
                stats["sf_rejected"] += 1
                continue
            reverse_found.add(move_type)

        if not reverse_found: return 0.0, set()

        stats["pawn_moves_ok"] += 1
        for mt in reverse_found:
            stats[f"reverse_{mt}"] += 1

        distinct = len(reverse_found)
        if distinct == 4:   score = 10.0
        elif distinct == 3: score = 4.0
        elif distinct == 2: score = 1.5
        elif distinct == 1: score = 0.3
        else:               score = 0.0

        return score, reverse_found

    def local_search(self, board, wP_sq, current_score, current_moves):
        global stats

        best_board = board.copy()
        best_score = current_score
        best_moves = current_moves.copy()
        improved = True

        wP_file = chess.square_file(wP_sq)
        wP_rank = chess.square_rank(wP_sq)
        sq_cL = chess.square(wP_file - 1, wP_rank + 1)
        sq_cR = chess.square(wP_file + 1, wP_rank + 1)
        forbidden = {wP_sq, sq_cL, sq_cR}

        def get_available(b):
            a = ['Q', 'R', 'B', 'N']
            if count_pieces(b, chess.WHITE, chess.QUEEN)  >= 1: a = [x for x in a if x != 'Q']
            if count_pieces(b, chess.WHITE, chess.ROOK)   >= 2: a = [x for x in a if x != 'R']
            if count_pieces(b, chess.WHITE, chess.BISHOP) >= 2: a = [x for x in a if x != 'B']
            if count_pieces(b, chess.WHITE, chess.KNIGHT) >= 2: a = [x for x in a if x != 'N']
            return a

        while improved and best_score < 10.0:
            improved = False

            # 1. Προσθήκη
            for sq in chess.SQUARES:
                if best_board.piece_at(sq): continue
                if sq in forbidden: continue
                for sym in get_available(best_board):
                    new_board = best_board.copy()
                    p_type = chess.Piece.from_symbol(sym).piece_type
                    new_board.set_piece_at(sq, chess.Piece(p_type, chess.WHITE))
                    new_board.turn = chess.WHITE
                    if not board_valid(new_board): continue
                    score, moves = self.evaluate_and_score(new_board, wP_sq)
                    if score > best_score:
                        best_score = score; best_board = new_board.copy()
                        best_moves = moves.copy(); improved = True
                        stats["local_search_improved"] += 1
                        print(f"    [+] {sym}@{chess.square_name(sq)}: {current_score}→{score} {sorted(moves)}")
                        if best_score == 10.0: return best_board, best_score, best_moves

            # 2. Αντικατάσταση
            for sq in chess.SQUARES:
                piece = best_board.piece_at(sq)
                if piece is None: continue
                if piece.color != chess.WHITE: continue
                if piece.piece_type in (chess.KING, chess.PAWN): continue
                for sym in get_available(best_board):
                    if chess.Piece.from_symbol(sym).piece_type == piece.piece_type: continue
                    new_board = best_board.copy()
                    p_type = chess.Piece.from_symbol(sym).piece_type
                    new_board.set_piece_at(sq, chess.Piece(p_type, chess.WHITE))
                    new_board.turn = chess.WHITE
                    if not board_valid(new_board): continue
                    score, moves = self.evaluate_and_score(new_board, wP_sq)
                    if score > best_score:
                        best_score = score; best_board = new_board.copy()
                        best_moves = moves.copy(); improved = True
                        stats["local_search_improved"] += 1
                        print(f"    [~] {piece.symbol()}→{sym}@{chess.square_name(sq)}: {current_score}→{score} {sorted(moves)}")
                        if best_score == 10.0: return best_board, best_score, best_moves

            # 3. Μετακίνηση
            for from_sq in chess.SQUARES:
                piece = best_board.piece_at(from_sq)
                if piece is None: continue
                if piece.color != chess.WHITE: continue
                if piece.piece_type in (chess.KING, chess.PAWN): continue
                for to_sq in chess.SQUARES:
                    if best_board.piece_at(to_sq): continue
                    if to_sq in forbidden: continue
                    if to_sq == from_sq: continue
                    new_board = best_board.copy()
                    new_board.remove_piece_at(from_sq)
                    new_board.set_piece_at(to_sq, piece)
                    new_board.turn = chess.WHITE
                    if not board_valid(new_board): continue
                    score, moves = self.evaluate_and_score(new_board, wP_sq)
                    if score > best_score:
                        best_score = score; best_board = new_board.copy()
                        best_moves = moves.copy(); improved = True
                        stats["local_search_improved"] += 1
                        print(f"    [→] {piece.symbol()} {chess.square_name(from_sq)}→{chess.square_name(to_sq)}: {current_score}→{score} {sorted(moves)}")
                        if best_score == 10.0: return best_board, best_score, best_moves

        return best_board, best_score, best_moves

    def construct_candidate(self):
        global stats
        stats["total"] += 1

        wP_sq = random.choice([chess.C2, chess.D2, chess.E2, chess.F2])
        wP_file = chess.square_file(wP_sq)
        wP_rank = 1

        sq_s1 = chess.square(wP_file,     wP_rank + 1)
        sq_s2 = chess.square(wP_file,     wP_rank + 2)
        sq_cL = chess.square(wP_file - 1, wP_rank + 1)
        sq_cR = chess.square(wP_file + 1, wP_rank + 1)

        board = chess.Board(None)
        board.set_piece_at(wP_sq, chess.Piece(chess.PAWN, chess.WHITE))
        occupied = {wP_sq}

        # Μαύρα κομμάτια στις διαγώνιες
        for cap_sq in [sq_cL, sq_cR]:
            sym = random.choice(['n', 'b', 'r', 'q'])
            p_type = chess.Piece.from_symbol(sym).piece_type
            board.set_piece_at(cap_sq, chess.Piece(p_type, chess.BLACK))
            occupied.add(cap_sq)

        # Μαύρος βασιλιάς
        bK_opts = []
        for df in range(2, 5):
            for ds in range(-2, 3):
                f = wP_file + ds
                r = wP_rank + df
                if 0 <= f <= 7 and 0 <= r <= 7:
                    sq = chess.square(f, r)
                    if sq not in occupied and sq not in {sq_s1, sq_s2}:
                        bK_opts.append(sq)
        if not bK_opts: return None, None

        bK_sq = random.choice(bK_opts)
        board.set_piece_at(bK_sq, chess.Piece(chess.KING, chess.BLACK))
        occupied.add(bK_sq)
        stats["geo_ok"] += 1

        # Λευκά κομμάτια — φυσικοί αριθμοί
        white_zone = [
            s for s in chess.SQUARES
            if s not in occupied
            and s not in {sq_cL, sq_cR}
        ]
        for _ in range(random.randint(2, 5)):
            opts = [s for s in white_zone if s not in occupied]
            if not opts: break

            available = ['Q', 'R', 'B', 'N']
            if count_pieces(board, chess.WHITE, chess.QUEEN)  >= 1: available = [x for x in available if x != 'Q']
            if count_pieces(board, chess.WHITE, chess.ROOK)   >= 2: available = [x for x in available if x != 'R']
            if count_pieces(board, chess.WHITE, chess.BISHOP) >= 2: available = [x for x in available if x != 'B']
            if count_pieces(board, chess.WHITE, chess.KNIGHT) >= 2: available = [x for x in available if x != 'N']
            if not available: break

            sym = random.choice(available)
            sq = choose_sq(opts, sym)
            if sq is None: break
            p_type = chess.Piece.from_symbol(sym).piece_type
            board.set_piece_at(sq, chess.Piece(p_type, chess.WHITE))
            occupied.add(sq)

        wK_sq = find_wK_safe(board)
        if wK_sq is None: return None, None
        board.set_piece_at(wK_sq, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.WHITE

        if not board_valid(board): return None, None
        stats["valid_board"] += 1
        return board, wP_sq


def run_reverse_albino_aco(iterations, ants_per_gen, stockfish_path):
    global pheromone_matrix, stats

    start_time = time.time()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    aco = ReverseAlbinoACO(engine)
    found_total = 0

    with open("reverse_albino_results.csv", mode='w',
              newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "score", "duration", "fen",
            "hero_pawn", "type", "variations"
        ])

        for gen in range(iterations):
            gen_data = []
            successes = 0
            print(f"\n=== Γενιά {gen+1}/{iterations} ===")

            for ant in range(ants_per_gen):
                board, wP_sq = aco.construct_candidate()
                if board is None: continue

                score, reverse_moves = aco.evaluate_and_score(board, wP_sq)

                if score >= 0.3:
                    print(f"\n  → Local search score={score} moves={sorted(reverse_moves)}...")
                    board, score, reverse_moves = aco.local_search(
                        board, wP_sq, score, reverse_moves)

                if score > 0:
                    gen_data.append((board.fen(), score))

                if score >= 4.0:
                    successes += 1
                    found_total += 1
                    elapsed = time.time() - start_time
                    duration_str = format_duration(elapsed)
                    pawn_str = chess.square_name(wP_sq)
                    fen = board.fen()

                    if score == 10.0: label = "ΤΕΛΕΙΟ REVERSE ALBINO (4/4)"
                    else:             label = "REVERSE ALBINO 3/4"

                    pawn_moves_dict = aco.get_pawn_moves(wP_sq)
                    variations_str = build_variations_string(
                        board, wP_sq, reverse_moves, pawn_moves_dict)

                    writer.writerow([
                        score, duration_str, fen,
                        pawn_str, label, variations_str
                    ])
                    file.flush()

                    print(f"\n  ✓ [{label}] #{found_total}")
                    print(f"  FEN:        {fen}")
                    print(f"  Hero pawn:  {pawn_str}")
                    print(f"  Κινήσεις:   {sorted(reverse_moves)}")
                    print(f"  Duration:   {duration_str}")
                    print(f"  Variations: {variations_str}")

            print(f"\n  → {successes} θέσεις αυτή τη γενιά.")
            print(
                f"  [STATS] total={stats['total']} geo={stats['geo_ok']} "
                f"valid={stats['valid_board']} no_pre={stats['no_premate']} "
                f"pawn_ok={stats['pawn_moves_ok']} "
                f"sf_rej={stats['sf_rejected']} "
                f"ls_ok={stats['local_search_improved']} | "
                f"1:{stats['reverse_1']} 2:{stats['reverse_2']} "
                f"3:{stats['reverse_3']} 4:{stats['reverse_4']}"
            )

            pheromone_matrix *= 0.95
            for fen_str, sc in gen_data:
                b = chess.Board(fen_str)
                for sq in chess.SQUARES:
                    p = b.piece_at(sq)
                    if p:
                        idx = PIECE_TO_INDEX[p.symbol()]
                        pheromone_matrix[sq][idx] += sc
            np.clip(pheromone_matrix, 0.01, 50.0, out=pheromone_matrix)

    engine.quit()
    total_seconds = time.time() - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    print(f"\n=== ΤΕΛΟΣ: Βρέθηκαν {found_total} θέσεις ===")
    print(f"Συνολικός χρόνος: {minutes} λεπτά και {seconds} δευτερόλεπτα")


if __name__ == "__main__":
    SF_PATH = r"C:\Users\Ektoras\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    run_reverse_albino_aco(
        iterations=50,
        ants_per_gen=500,
        stockfish_path=SF_PATH
    )