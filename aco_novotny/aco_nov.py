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
    "total": 0, "geo_ok": 0, "threat_found": 0, "def_ok": 0,
    "no_premate": 0, "legal": 0, "forced": 0,
    "caps": 0, "m1": 0, "m2d": 0, "m2p": 0,
}

def knight_destinations(from_sq):
    f, r = chess.square_file(from_sq), chess.square_rank(from_sq)
    moves = []
    for df, dr in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nf, nr = f + df, r + dr
        if 0 <= nf <= 7 and 0 <= nr <= 7:
            moves.append(chess.square(nf, nr))
    return moves

def get_rook_line(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return [s for s in chess.SQUARES if s != sq and
            (chess.square_file(s) == f or chess.square_rank(s) == r)]

def get_bishop_line(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return [s for s in chess.SQUARES if s != sq and
            abs(chess.square_file(s) - f) == abs(chess.square_rank(s) - r)]

def choose_sq(options, piece_symbol):
    idx = PIECE_TO_INDEX[piece_symbol]
    values = np.array([pheromone_matrix[sq][idx] for sq in options])
    total = values.sum()
    if total == 0:
        return random.choice(options)
    probs = values / total
    return np.random.choice(options, p=probs)

def find_wK_safe(board):
    bk_sq = board.king(chess.BLACK)
    candidates = []
    for sq in chess.SQUARES:
        if board.piece_at(sq) is not None:
            continue
        if bk_sq is not None and chess.square_distance(sq, bk_sq) <= 1:
            continue
        board.set_piece_at(sq, chess.Piece(chess.KING, chess.WHITE))
        board.turn = chess.WHITE
        if not board.is_check():
            candidates.append(sq)
        board.remove_piece_at(sq)
    return random.choice(candidates) if candidates else None

def board_valid(board):
    try:
        return board.is_valid()
    except Exception:
        return False

def get_mate1_move(engine, board):
    res = engine.analyse(board, chess.engine.Limit(depth=4))
    sc = res["score"].white()
    if sc.is_mate() and sc.mate() == 1:
        play = engine.play(board, chess.engine.Limit(depth=4))
        return play.move
    return None

def format_duration(seconds):
    """Μετατρέπει δευτερόλεπτα σε μορφή '2m 14s'."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def build_variations_string(board_after_sacrifice, end_sq, rook_cap, bish_cap, engine):
    """
    Φτιάχνει το variations string με μορφή:
    RxS:mate_move | BxS:mate_move
    π.χ. "Re5xd3:Rg3d3 | Be6xd3:Bh7d3"
    """
    parts = []
    labels = [("RxS", rook_cap), ("BxS", bish_cap)]
    for label, cap in labels:
        if cap is None:
            parts.append(f"{label}:???")
            continue
        tb = board_after_sacrifice.copy()
        tb.push(cap)
        m = get_mate1_move(engine, tb)
        cap_uci = cap.uci()
        mate_uci = m.uci() if m else "???"
        parts.append(f"{cap_uci}:{mate_uci}")
    return " | ".join(parts)


class NovotnyACO:
    def __init__(self, engine):
        self.engine = engine

    def construct_candidate(self):
        global stats
        stats["total"] += 1

        S = random.choice(list(chess.SQUARES))
        rook_line = get_rook_line(S)
        bish_line = get_bishop_line(S)
        if not rook_line or not bish_line:
            return None, None, None

        bR_sq = choose_sq(rook_line, 'r')
        bish_free = [s for s in bish_line if s != bR_sq]
        if not bish_free:
            return None, None, None
        bB_sq = choose_sq(bish_free, 'b')

        bk_opts = [s for s in chess.SQUARES
                   if s not in (S, bR_sq, bB_sq)
                   and chess.square_distance(s, S) <= 3]
        if not bk_opts:
            return None, None, None
        bK_sq = random.choice(bk_opts)
        stats["geo_ok"] += 1

        best_white = None
        mate_RxS = None
        mate_BxS = None

        for attempt in range(40):
            board_base = chess.Board(None)
            board_base.set_piece_at(bK_sq, chess.Piece(chess.KING,   chess.BLACK))
            board_base.set_piece_at(bR_sq, chess.Piece(chess.ROOK,   chess.BLACK))
            board_base.set_piece_at(bB_sq, chess.Piece(chess.BISHOP, chess.BLACK))

            occupied = {bK_sq, bR_sq, bB_sq, S}
            white_pieces = {}

            for _ in range(random.randint(2, 4)):
                opts = [s for s in chess.SQUARES
                        if s not in occupied
                        and chess.square_distance(s, bK_sq) <= 5]
                if not opts:
                    break
                pt = random.choice([chess.ROOK, chess.BISHOP,
                                    chess.QUEEN, chess.KNIGHT])
                sq = choose_sq(opts, chess.piece_symbol(pt).upper())
                board_base.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
                white_pieces[sq] = pt
                occupied.add(sq)

            wK_sq = find_wK_safe(board_base)
            if wK_sq is None:
                continue
            board_base.set_piece_at(wK_sq, chess.Piece(chess.KING, chess.WHITE))
            white_pieces[wK_sq] = chess.KING
            board_base.turn = chess.WHITE
            if not board_valid(board_base):
                continue

            pre = self.engine.analyse(board_base, chess.engine.Limit(depth=4))
            pre_sc = pre["score"].white()
            if pre_sc.is_mate() and pre_sc.mate() == 1:
                continue

            b_RxS = chess.Board(None)
            b_RxS.set_piece_at(bK_sq, chess.Piece(chess.KING,   chess.BLACK))
            b_RxS.set_piece_at(S,     chess.Piece(chess.ROOK,   chess.BLACK))
            b_RxS.set_piece_at(bB_sq, chess.Piece(chess.BISHOP, chess.BLACK))
            conflict = any(b_RxS.piece_at(sq) for sq in white_pieces)
            if conflict:
                continue
            for sq, pt in white_pieces.items():
                b_RxS.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
            b_RxS.turn = chess.WHITE
            if not board_valid(b_RxS):
                continue
            m_R = get_mate1_move(self.engine, b_RxS)
            if m_R is None:
                continue

            b_BxS = chess.Board(None)
            b_BxS.set_piece_at(bK_sq, chess.Piece(chess.KING,   chess.BLACK))
            b_BxS.set_piece_at(bR_sq, chess.Piece(chess.ROOK,   chess.BLACK))
            b_BxS.set_piece_at(S,     chess.Piece(chess.BISHOP, chess.BLACK))
            conflict = any(b_BxS.piece_at(sq) for sq in white_pieces)
            if conflict:
                continue
            for sq, pt in white_pieces.items():
                b_BxS.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
            b_BxS.turn = chess.WHITE
            if not board_valid(b_BxS):
                continue
            m_B = get_mate1_move(self.engine, b_BxS)
            if m_B is None:
                continue

            if m_R == m_B:
                continue

            best_white = white_pieces
            mate_RxS = m_R
            mate_BxS = m_B
            break

        if best_white is None:
            return None, None, None
        stats["threat_found"] += 1

        full = chess.Board(None)
        full.set_piece_at(bK_sq, chess.Piece(chess.KING,   chess.BLACK))
        full.set_piece_at(bR_sq, chess.Piece(chess.ROOK,   chess.BLACK))
        full.set_piece_at(bB_sq, chess.Piece(chess.BISHOP, chess.BLACK))

        occupied_f = {bK_sq, bR_sq, bB_sq, S}
        for sq, pt in best_white.items():
            if sq not in occupied_f:
                full.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
                occupied_f.add(sq)

        if full.king(chess.WHITE) is None:
            wK_sq = find_wK_safe(full)
            if wK_sq is None:
                return None, None, None
            full.set_piece_at(wK_sq, chess.Piece(chess.KING, chess.WHITE))
            occupied_f.add(wK_sq)

        n_origins = [sq for sq in chess.SQUARES
                     if S in knight_destinations(sq) and sq not in occupied_f]
        if not n_origins:
            return None, None, None
        origin_sq = choose_sq(n_origins, 'N')
        full.set_piece_at(origin_sq, chess.Piece(chess.KNIGHT, chess.WHITE))

        full.turn = chess.WHITE
        if not board_valid(full):
            return None, None, None

        stats["def_ok"] += 1
        return full, origin_sq, S

    def evaluate_and_score(self, board, start_sq, end_sq):
        global stats
        if board is None:
            return 0, None

        bc = board.copy()
        bc.turn = chess.WHITE

        pre = self.engine.analyse(bc, chess.engine.Limit(depth=5))
        if pre["score"].white().is_mate() and pre["score"].white().mate() == 1:
            return 0, None
        stats["no_premate"] += 1

        move = chess.Move(start_sq, end_sq)
        if move not in bc.legal_moves:
            return 0, None
        stats["legal"] += 1

        bc.push(move)

        saves_black = False
        non_caps = [m for m in bc.legal_moves if m.to_square != end_sq]
        for bm in non_caps:
            tb = bc.copy()
            tb.push(bm)
            res = self.engine.analyse(tb, chess.engine.Limit(depth=3))
            sc = res["score"].white()
            if not (sc.is_mate() and sc.mate() == 1):
                saves_black = True
                break

        if saves_black:
            bc.pop()
            return 0, None
        stats["forced"] += 1

        rook_cap = None
        bish_cap = None
        for cap in bc.legal_moves:
            if cap.to_square != end_sq:
                continue
            p = bc.piece_at(cap.from_square)
            if p and p.color == chess.BLACK:
                if p.piece_type == chess.ROOK:
                    rook_cap = cap
                elif p.piece_type == chess.BISHOP:
                    bish_cap = cap

        if rook_cap is None or bish_cap is None:
            bc.pop()
            return 0.05, None
        stats["caps"] += 1

        mate_moves = []
        mate_count = 0
        for cap in [rook_cap, bish_cap]:
            tb = bc.copy()
            tb.push(cap)
            res = self.engine.analyse(tb, chess.engine.Limit(depth=3))
            sc = res["score"].white()
            if sc.is_mate() and sc.mate() == 1:
                play = self.engine.play(tb, chess.engine.Limit(depth=3))
                mate_moves.append(play.move)
                mate_count += 1

        # Φτιάχνουμε το variations string
        variations = build_variations_string(bc, end_sq, rook_cap, bish_cap, self.engine)
        bc.pop()

        if mate_count == 2:
            if mate_moves[0] != mate_moves[1]:
                stats["m2p"] += 1
                return 5.0, variations
            else:
                stats["m2d"] += 1
                return 3.0, variations
        elif mate_count == 1:
            stats["m1"] += 1
            return 1.0, variations
        return 0.1, None


def run_aco_to_csv(iterations, ants_per_gen, stockfish_path):
    global pheromone_matrix, stats

    start_time = time.time()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    aco = NovotnyACO(engine)
    found_total = 0

    with open("novotny_results.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "score", "duration", "fen",
                "key_move", "type", "variations"
            ])

        for gen in range(iterations):
            gen_data = []
            successes = 0
            print(f"\n=== Γενιά {gen+1}/{iterations} ===")

            for ant in range(ants_per_gen):
                board, start, end = aco.construct_candidate()
                if board is None:
                    continue
                score, variations = aco.evaluate_and_score(board, start, end)
                if score > 0:
                    gen_data.append((board.fen(), score))
                if score >= 3.0:
                    successes += 1
                    found_total += 1
                    elapsed = time.time() - start_time
                    duration_str = format_duration(elapsed)
                    move_str = f"{chess.square_name(start)}-{chess.square_name(end)}"
                    fen = board.fen()
                    label = "PERFECT" if score == 5.0 else "DUALITY"

                    writer.writerow([
                        score, duration_str, fen,
                        move_str, label, variations
                    ])
                    file.flush()
                    print(f"  YES [{label}] #{found_total}: {move_str} | {fen}")
                    print(f"     variations: {variations}")
                    print(f"     duration:   {duration_str}")

            print(f"  → {successes} Novotny σε αυτή τη γενιά.")
            print(
                f"  [DEBUG] total={stats['total']} geo={stats['geo_ok']} "
                f"threat={stats['threat_found']} def={stats['def_ok']} "
                f"pre={stats['no_premate']} leg={stats['legal']} "
                f"forced={stats['forced']} caps={stats['caps']} "
                f"m1={stats['m1']} m2d={stats['m2d']} m2p={stats['m2p']}"
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
    print(f"\n=== ΤΕΛΟΣ: {found_total} Novotny αποθηκεύτηκαν ===")
    print(f"Συνολικός χρόνος εκτέλεσης: {minutes} λεπτά και {seconds} δευτερόλεπτα")


if __name__ == "__main__":
    SF_PATH = r"C:\Users\Ektoras\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    run_aco_to_csv(
        iterations=30,
        ants_per_gen=300,
        stockfish_path=SF_PATH
    )