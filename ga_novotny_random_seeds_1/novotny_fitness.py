import chess
import chess.engine
# =============== MATE-IN-1 HELPER (ΜΟΝΑΔΙΚΟΤΗΤΑ) ===============

def unique_white_mate_in_1(board: chess.Board):
    """Επιστρέφει μία κίνηση #1 αν είναι ΜΟΝΑΔΙΚΗ, αλλιώς None."""
    if board.turn != chess.WHITE:
        return None

    best = None
    count = 0

    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            count += 1
            if count == 1:
                best = m
            else:
                board.pop()
                return None
        board.pop()

    return best if count == 1 else None


# =============== ΕΥΡΕΣΗ ΥΠΟΨΗΦΙΟΥ HERO-MOVE ΓΙΑ NOVOTNY ===============

def find_candidate_novotny_hero(board: chess.Board):
    best_move = None
    best_local_score = -10**9

    for m in board.legal_moves:
        piece = board.piece_at(m.from_square)
        if piece is None or piece.color != chess.WHITE:
            continue
        if piece.piece_type == chess.KING:
            continue
        if m.promotion is not None:
            continue

        v = m.to_square

        rook_like_before = 0
        bishop_like_before = 0
        for sq in board.attackers(chess.BLACK, v):
            att = board.piece_at(sq)
            if att is None or att.color != chess.BLACK:
                continue
            df = chess.square_file(v) - chess.square_file(sq)
            dr = chess.square_rank(v) - chess.square_rank(sq)
            if att.piece_type in (chess.ROOK, chess.QUEEN) and (df == 0 or dr == 0):
                rook_like_before += 1
            if att.piece_type in (chess.BISHOP, chess.QUEEN) and abs(df) == abs(dr):
                bishop_like_before += 1

        # ΣΚΛΗΡΟ ΦΙΛΤΡΟ: απαιτούμε και τους δύο τύπους επίθεσης
        if rook_like_before == 0 or bishop_like_before == 0:
            continue

        b2 = board.copy()
        b2.push(m)

        rook_like_caps = 0
        bishop_like_caps = 0
        total_caps = 0

        for d in b2.legal_moves:
            if d.to_square != v:
                continue
            attacker = b2.piece_at(d.from_square)
            if attacker is None or attacker.color != chess.BLACK:
                continue

            df = chess.square_file(d.to_square) - chess.square_file(d.from_square)
            dr = chess.square_rank(d.to_square) - chess.square_rank(d.from_square)

            is_rook_line   = (attacker.piece_type in (chess.ROOK, chess.QUEEN)) and (df == 0 or dr == 0)
            is_bishop_line = (attacker.piece_type in (chess.BISHOP, chess.QUEEN)) and (abs(df) == abs(dr))

            if is_rook_line or is_bishop_line:
                total_caps += 1
            if is_rook_line:
                rook_like_caps += 1
            if is_bishop_line:
                bishop_like_caps += 1

        local_score = 150
        local_score += 20 * min(rook_like_before, 2)
        local_score += 20 * min(bishop_like_before, 2)

        if total_caps >= 2:
            local_score += 100
        elif total_caps == 1:
            local_score += 30
        else:
            local_score -= 50

        if rook_like_caps > 0 and bishop_like_caps > 0:
            local_score += 120

        if local_score > best_local_score:
            best_local_score = local_score
            best_move = m

    return best_move


# =============== SOFT NOVOTNY FITNESS ===============

def evaluate_soft_novotny(board: chess.Board, engine: chess.engine.SimpleEngine):

    if not board.is_valid():
        return -1000.0, None, {"reason": "invalid_board"}

    if board.is_check():
        return -800.0, None, {"reason": "side_to_move_in_check"}

    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return -1000.0, None, {"reason": "missing_king"}

    # === [ΕΛΕΓΧΟΣ ΥΠΑΡΞΗΣ ΜΑΥΡΟΥ ΠΥΡΓΟΥ ΚΑΙ ΑΞΙΩΜΑΤΙΚΟΥ] ===
    # Χωρίς αυτά τα δύο κομμάτια το Novotny είναι αδύνατο
    black_pieces = [p for p in board.piece_map().values() if p.color == chess.BLACK]
    black_types = [p.piece_type for p in black_pieces]
    if chess.ROOK not in black_types or chess.BISHOP not in black_types:
        return -2000.0, None, {"reason": "missing_black_rook_or_bishop"}

    # === [ΠΡΩΙΜΟΣ ΕΛΕΓΧΟΣ: Η ΘΕΣΗ ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ ΚΕΡΔΙΣΜΕΝΗ ΓΙΑ ΤΟΝ ΛΕΥΚΟ] ===
    try:
        info_quick = engine.analyse(board, chess.engine.Limit(depth=8), multipv=1)
        if info_quick:
            quick_score = info_quick[0]["score"].pov(chess.WHITE)
            quick_mate = quick_score.mate()
            quick_cp = quick_score.score(mate_score=100000)

            if quick_mate is not None and quick_mate < 0:
                return -3000.0, None, {"reason": "black_has_mate"}
            if quick_cp is not None and quick_cp < 0:
                return -2500.0, None, {"reason": "black_is_winning"}
    except Exception:
        pass

    if unique_white_mate_in_1(board) is not None:
        return -700.0, None, {"reason": "already_unique_mate_in_1"}

    for m in board.legal_moves:
        board.push(m)
        is_m1 = board.is_checkmate()
        board.pop()
        if is_m1:
            return -1000.0, None, {"reason": "has_any_mate_in_1"}

    piece_map = board.piece_map()
    material_count = len(piece_map)

    penalty = 0.0

    if material_count > 14:
        penalty -= 40.0 * (material_count - 14)
    if material_count > 20:
        penalty -= 200.0

    white_heavy = 0
    black_heavy = 0
    for p in piece_map.values():
        if p.piece_type in (chess.QUEEN, chess.ROOK):
            if p.color == chess.WHITE:
                white_heavy += 1
            else:
                black_heavy += 1

    if white_heavy > 2:
        penalty -= 60.0 * (white_heavy - 2)
    if black_heavy > 2:
        penalty -= 40.0 * (black_heavy - 2)

    # ---- Hero move ----
    hero_move = find_candidate_novotny_hero(board)
    if hero_move is None:
        return -500.0 + penalty, None, {"reason": "no_candidate_hero"}

    v = hero_move.to_square

    score = 0.0
    details = {
        "hero_uci": hero_move.uci(),
        "square_v": chess.square_name(v),
        "rook_like_before": 0,
        "bishop_like_before": 0,
        "rook_line_caps": 0,
        "bishop_line_caps": 0,
        "total_caps": 0,
        "unique_mates_after_caps": 0,
        "white_heavy": white_heavy,
        "black_heavy": black_heavy,
    }

    b_hero = board.copy()
    b_hero.push(hero_move)
    if b_hero.is_checkmate():
        return -400.0 + penalty, hero_move, {"reason": "hero_is_mate_in_1"}
    if b_hero.is_check():
        score -= 150.0

    rook_like_before = 0
    bishop_like_before = 0

    for sq in board.attackers(chess.BLACK, v):
        att = board.piece_at(sq)
        if att is None or att.color != chess.BLACK:
            continue
        df = chess.square_file(v) - chess.square_file(sq)
        dr = chess.square_rank(v) - chess.square_rank(sq)
        if att.piece_type in (chess.ROOK, chess.QUEEN) and (df == 0 or dr == 0):
            rook_like_before += 1
        if att.piece_type in (chess.BISHOP, chess.QUEEN) and abs(df) == abs(dr):
            bishop_like_before += 1

    details["rook_like_before"] = rook_like_before
    details["bishop_like_before"] = bishop_like_before

    total_attackers = rook_like_before + bishop_like_before

    if rook_like_before == 0 and bishop_like_before == 0:
        return -300.0 + penalty, hero_move, {"reason": "no_RB_attackers"}
    elif rook_like_before > 0 and bishop_like_before > 0:
        score += 240.0
        score += 25.0 * min(rook_like_before, 2)
        score += 25.0 * min(bishop_like_before, 2)
        if total_attackers > 4:
            score -= 25.0 * (total_attackers - 4)
    else:
        score += 40.0

    # === [ΥΠΟΛΟΓΙΣΜΟΣ ΑΜΥΝΩΝ & ΜΑΤ] ===
    b2 = board.copy()
    b2.push(hero_move)

    rook_line_caps = 0
    bishop_line_caps = 0
    total_caps = 0
    unique_mates_after_caps = 0
    defense_str_list = []
    unique_mate_moves = set()

    for d in b2.legal_moves:
        if d.to_square != v:
            continue

        attacker = b2.piece_at(d.from_square)
        if attacker is None or attacker.color != chess.BLACK:
            continue

        df = chess.square_file(d.to_square) - chess.square_file(d.from_square)
        dr = chess.square_rank(d.to_square) - chess.square_rank(d.from_square)
        is_rook_line   = (attacker.piece_type in (chess.ROOK, chess.QUEEN)) and (df == 0 or dr == 0)
        is_bishop_line = (attacker.piece_type in (chess.BISHOP, chess.QUEEN)) and abs(df) == abs(dr)

        if not (is_rook_line or is_bishop_line):
            continue

        total_caps += 1
        if is_rook_line: rook_line_caps += 1
        if is_bishop_line: bishop_line_caps += 1

        b2.push(d)
        m1 = unique_white_mate_in_1(b2)
        if m1 is not None:
            unique_mates_after_caps += 1
            defense_str_list.append(f"{d.uci()}:{m1.uci()}")
            unique_mate_moves.add(m1.uci())
        b2.pop()

    details["defenses"] = " | ".join(defense_str_list)
    details["unique_mates_after_caps"] = unique_mates_after_caps

    # === [SCORING ΓΙΑ NOVOTNY THEME] ===
    if unique_mates_after_caps >= 2:
        if len(unique_mate_moves) >= 2:
            score += 3000.0  # Τέλειο Novotny: διαφορετικό ματ για κάθε άμυνα
        else:
            # Πολλές αμυντικές κινήσεις αλλά ΙΔΙΟ ματ — δεν είναι πραγματικό Novotny
            score -= 500.0
    elif unique_mates_after_caps == 1:
        score += 150.0
    elif unique_mates_after_caps == 0 and total_caps > 0:
        score -= 200.0

    if rook_line_caps > 0 and bishop_line_caps > 0:
        score += 300.0

    # === [STOCKFISH ANALYSIS - ΜΟΝΑΔΙΚΟΤΗΤΑ ΛΥΣΗΣ] ===
    SF_DEPTH = 18
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=2)

        best_move_sf = None
        hero_eval = -100000
        hero_mate_depth = None

        if len(info) > 0:
            best_move_sf = info[0]["pv"][0]
            sc0 = info[0]["score"].pov(chess.WHITE)
            hero_eval = sc0.score(mate_score=100000)
            hero_mate_depth = sc0.mate()

        if hero_move != best_move_sf:
            return -2000.0, None, {"reason": "hero_not_top_move"}

        if hero_mate_depth is not None:
            if hero_mate_depth == 2:
                score += 2000.0
            elif hero_mate_depth > 2:
                score += 500.0
        else:
            if hero_eval > 500:
                score += 200.0
            else:
                return -1500.0, None, {"reason": "no_mate_found_by_sf"}

        details["stockfish_cp_score"] = hero_eval
        details["top_move_sf"] = best_move_sf.uci() if best_move_sf else "None"

    except Exception as e:
        score -= 100.0
        details["sf_error"] = str(e)

    score += penalty
    return score, hero_move, details