import chess
import chess.engine
import random

# ==================== MATE-IN-1 HELPER ====================

def unique_white_mate_in_1(board: chess.Board):
    """Επιστρέφει μία κίνηση ματ-σε-1 αν είναι ΜΟΝΑΔΙΚΗ, αλλιώς None."""
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


# ==================== MATE-IN-2 HELPER ====================

def white_has_mate_in_2(board: chess.Board):
    """
    Ελέγχει αν ο Λευκός έχει ματ-σε-2 (brute force).
    Επιστρέφει True αν για ΚΑΘΕ κίνηση του Μαύρου υπάρχει ματ-σε-1 για τον Λευκό.
    """
    if board.turn != chess.WHITE:
        return False

    for w_move in board.legal_moves:
        board.push(w_move)
        # Τώρα παίζει ο Μαύρος
        black_can_escape = False
        for b_move in board.legal_moves:
            board.push(b_move)
            # Τώρα παίζει ο Λευκός — ελέγχουμε αν έχει ματ-σε-1
            has_m1 = False
            for w2 in board.legal_moves:
                board.push(w2)
                if board.is_checkmate():
                    has_m1 = True
                board.pop()
                if has_m1:
                    break
            if not has_m1:
                black_can_escape = True
            board.pop()
            if black_can_escape:
                break
        board.pop()
        if not black_can_escape:
            return True  # Βρήκαμε κίνηση που δίνει ματ-σε-2
    return False


# ==================== ΕΥΡΕΣΗ HERO-MOVE (AUW) ====================

def find_candidate_auw_hero(board: chess.Board):
    """
    Αναζητά κίνηση (Hero Move) που αφορά πιόνι στην 6η ή 7η σειρά.
    Προτιμάμε κινήσεις που φέρνουν πιόνι στην 7η σειρά (rank 6).
    """
    candidates_rank6 = []  # Κινήσεις που φέρνουν πιόνι ακριβώς στην 7η
    candidates_other = []  # Άλλες κινήσεις με πιόνι ήδη στην 7η

    for m in board.legal_moves:
        piece = board.piece_at(m.from_square)
        if piece is None or piece.piece_type == chess.KING:
            continue

        b2 = board.copy()
        b2.push(m)

        # Έλεγχος αν υπάρχει πιόνι στην 7η σειρά μετά την κίνηση
        pawn_on_7th = None
        for sq in chess.SQUARES:
            p = b2.piece_at(sq)
            if p and p.color == chess.WHITE and p.piece_type == chess.PAWN and chess.square_rank(sq) == 6:
                pawn_on_7th = sq
                break

        if pawn_on_7th is None:
            continue

        # Προτιμάμε κινήσεις όπου το ίδιο το πιόνι μετακινήθηκε στην 7η
        if (piece.piece_type == chess.PAWN and
                chess.square_rank(m.from_square) == 5 and
                chess.square_rank(m.to_square) == 6):
            candidates_rank6.append(m)
        else:
            candidates_other.append(m)

    if candidates_rank6:
        return random.choice(candidates_rank6)
    if candidates_other:
        return random.choice(candidates_other)
    return None


# ==================== AUW FITNESS ====================

def evaluate_auw_fitness(board: chess.Board, engine: chess.engine.SimpleEngine):
    score = 0.0
    penalty = 0.0
    SF_DEPTH = 14

    # ===== ΒΑΣΙΚΟΙ ΕΛΕΓΧΟΙ =====

    if not board.is_valid():
        return -1000.0, None, {"reason": "invalid_board"}

    if board.is_check():
        return -800.0, None, {"reason": "in_check"}

    # Οικονομία: penalty για πολλά κομμάτια
    piece_count = len(board.piece_map())
    if piece_count > 12:
        penalty -= (piece_count - 12) * 35.0

    # Penalty αν υπάρχει ήδη ματ-σε-1
    if unique_white_mate_in_1(board) is not None:
        return -1500.0 + penalty, None, {"reason": "already_mate_in_1"}

    # ===== STOCKFISH: Η ΑΡΧΙΚΗ ΘΕΣΗ ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ MATE-IN-3 =====
    try:
        info_initial = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=1)
        if info_initial:
            sc_initial = info_initial[0]["score"].pov(chess.WHITE)
            mate_initial = sc_initial.mate()

            if mate_initial is None:
                # Δεν είναι ματ καθόλου
                return -3000.0 + penalty, None, {"reason": "no_mate_found"}
            if mate_initial != 3:
                # Είναι ματ αλλά όχι σε 3
                return -2000.0 + penalty, None, {"reason": f"mate_in_{mate_initial}_not_3"}
    except Exception as e:
        return -1000.0, None, {"reason": f"sf_initial_error: {e}"}

    # ===== ΕΥΡΕΣΗ HERO MOVE =====
    hero_move = find_candidate_auw_hero(board)
    if hero_move is None:
        return -500.0 + penalty, None, {"reason": "no_candidate_hero"}

    b_hero = board.copy()
    b_hero.push(hero_move)

    if b_hero.is_checkmate():
        return -2000.0 + penalty, hero_move, {"reason": "hero_is_checkmate"}

    # Βρίσκουμε το πιόνι στην 7η σειρά μετά το hero move
    p_hero_sq = None
    for sq in chess.SQUARES:
        p = b_hero.piece_at(sq)
        if p and p.color == chess.WHITE and p.piece_type == chess.PAWN and chess.square_rank(sq) == 6:
            p_hero_sq = sq
            break

    if p_hero_sq is None:
        return -400.0 + penalty, hero_move, {"reason": "no_pawn_on_7th_after_hero"}

    # ===== STOCKFISH: ΜΕΤΑ ΤΟ HERO MOVE, Ο ΜΑΥΡΟΣ ΠΡΕΠΕΙ ΝΑ ΤΡΩΕΙ ΜΑΤ ΜΑΧ ΣΕ 2 =====
    # ΚΑΙ Η PV ΠΡΕΠΕΙ ΝΑ ΠΕΡΙΛΑΜΒΑΝΕΙ ΠΡΟΑΓΩΓΗ ΤΟΥ HERO PAWN
    sf_pv_promotion = None  # τι προαγωγή προτείνει το Stockfish (αν υπάρχει)
    try:
        info_hero = engine.analyse(b_hero, chess.engine.Limit(depth=SF_DEPTH), multipv=1)
        if info_hero:
            sc_hero = info_hero[0]["score"].pov(chess.BLACK)
            mate_after_hero = sc_hero.mate()

            if mate_after_hero is None or mate_after_hero > 0:
                # Ο Μαύρος δεν τρώει ματ ή κερδίζει
                return -4000.0 + penalty, hero_move, {"reason": "black_survives_after_hero"}
            if mate_after_hero < -2:
                # Ο Μαύρος μπορεί να αντισταθεί πέρα από 2 κινήσεις
                return -3500.0 + penalty, hero_move, {"reason": f"black_survives_{abs(mate_after_hero)}_moves"}

            # Έλεγχος: η PV περιλαμβάνει προαγωγή του p_hero_sq;
            pv_moves = info_hero[0].get("pv", [])
            for pv_move in pv_moves:
                if pv_move.from_square == p_hero_sq and pv_move.promotion is not None:
                    sf_pv_promotion = pv_move.promotion
                    break

            if sf_pv_promotion is None:
                # Η καλύτερη γραμμή του Stockfish δεν περνάει από προαγωγή του hero pawn
                return -3000.0 + penalty, hero_move, {"reason": "sf_best_line_no_promotion"}
    except Exception as e:
        return -1000.0, hero_move, {"reason": f"sf_hero_error: {e}"}

    # ===== BRUTE FORCE: ΕΛΕΓΧΟΣ ΟΛΩΝ ΤΩΝ ΑΜΥΝΩΝ ΤΟΥ ΜΑΥΡΟΥ =====
    # Για κάθε νόμιμη κίνηση του Μαύρου μετά το hero move,
    # ελέγχουμε αν υπάρχει προαγωγή που δίνει ματ (άμεσα ή σε 1 ακόμα κίνηση)

    achieved_promos = {}   # promo_type -> (defense_uci, mate_uci)
    dual_defenses = []     # άμυνες που επιτρέπουν πολλαπλές προαγωγές
    auw_solutions = []
    all_defenses_covered = True  # Αν ΟΛΕΣ οι άμυνες καλύπτονται από κάποιο ματ

    for defense_move in b_hero.legal_moves:
        b_def = b_hero.copy()
        b_def.push(defense_move)

        # Βρίσκουμε ποιες προαγωγές δίνουν ματ (άμεσα ή σε 1)
        valid_promos_for_defense = []

        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            for move in b_def.legal_moves:
                if move.from_square != p_hero_sq or move.promotion != promo:
                    continue

                b_promo = b_def.copy()
                b_promo.push(move)

                # Ελέγχουμε ματ άμεσα
                if b_promo.is_checkmate():
                    valid_promos_for_defense.append((promo, move.uci(), "mate_in_1"))
                    break

                # Ελέγχουμε ματ σε 1 ακόμα κίνηση (ο Μαύρος παίζει, μετά ματ)
                if b_promo.turn == chess.BLACK:
                    black_can_escape = False
                    for bm in b_promo.legal_moves:
                        b_promo2 = b_promo.copy()
                        b_promo2.push(bm)
                        if unique_white_mate_in_1(b_promo2) is None:
                            black_can_escape = True
                            break
                    if not black_can_escape:
                        valid_promos_for_defense.append((promo, move.uci(), "mate_in_2"))
                        break

        if not valid_promos_for_defense:
            # Αυτή η άμυνα δεν καλύπτεται από καμία προαγωγή — πρόβλημα!
            all_defenses_covered = False
            penalty -= 500.0
            continue

        if len(valid_promos_for_defense) == 1:
            # Μοναδική προαγωγή για αυτή την άμυνα — αυτό θέλουμε!
            promo_type, mate_uci, depth_str = valid_promos_for_defense[0]
            p_name = {chess.QUEEN: "Queen", chess.ROOK: "Rook",
                      chess.BISHOP: "Bishop", chess.KNIGHT: "Knight"}.get(promo_type)

            if promo_type not in achieved_promos:
                achieved_promos[promo_type] = (defense_move.uci(), mate_uci)
                auw_solutions.append(
                    f"Defense:{defense_move.uci()} -> Mate:{mate_uci} ONLY {p_name} ({depth_str})"
                )
        else:
            # DUAL — πολλαπλές προαγωγές για αυτή την άμυνα
            dual_defenses.append(defense_move.uci())
            all_promos_str = [
                {chess.QUEEN: "Q", chess.ROOK: "R",
                 chess.BISHOP: "B", chess.KNIGHT: "N"}.get(p) for p, _, _ in valid_promos_for_defense
            ]
            auw_solutions.append(
                f"Defense:{defense_move.uci()} -> DUAL ({','.join(all_promos_str)})"
            )
            penalty -= 200.0  # Penalty για dual

    # ===== SCORING =====

    num_unique_promos = len(achieved_promos)
    promotions_found = set(achieved_promos.keys())

    # Bonus ανά μοναδική προαγωγή
    promo_bonus = {
        chess.KNIGHT: 3000.0,
        chess.ROOK:   2000.0,
        chess.BISHOP: 2000.0,
        chess.QUEEN:   500.0,
    }
    for p in promotions_found:
        score += promo_bonus.get(p, 0)

    # Bonus για αριθμό μοναδικών προαγωγών
    if num_unique_promos == 0:
        score -= 3000.0
    elif num_unique_promos == 1:
        score -= 1000.0
    elif num_unique_promos == 2:
        score += 1000.0
    elif num_unique_promos == 3:
        score += 5000.0
    elif num_unique_promos == 4:
        score += 12000.0  # AUW επιτεύχθηκε!

    # Bonus αν όλες οι άμυνες καλύπτονται
    if all_defenses_covered:
        score += 1000.0

    # Bonus για underpromotions
    has_underpromo = any(p in [chess.KNIGHT, chess.ROOK, chess.BISHOP] for p in promotions_found)
    if has_underpromo:
        score += 800.0

    # Penalty για duals
    penalty -= len(dual_defenses) * 150.0

    # Bonus αν η top γραμμή του Stockfish περνάει από μια από τις "ONLY" προαγωγές μας
    sf_pv_matches_solution = sf_pv_promotion in promotions_found
    if sf_pv_matches_solution:
        score += 1500.0

    total_fitness = score + penalty

    sf_pv_promo_name = {chess.QUEEN: "Q", chess.ROOK: "R",
                         chess.BISHOP: "B", chess.KNIGHT: "N"}.get(sf_pv_promotion, "None")

    details = {
        "hero_uci": hero_move.uci(),
        "p_hero_sq": chess.square_name(p_hero_sq),
        "auw_found": num_unique_promos,
        "promotions_list": [
            {chess.QUEEN: "Q", chess.ROOK: "R",
             chess.BISHOP: "B", chess.KNIGHT: "N"}.get(p) for p in promotions_found
        ],
        "solutions": auw_solutions,
        "dual_defenses": dual_defenses,
        "all_defenses_covered": all_defenses_covered,
        "sf_pv_promotion": sf_pv_promo_name,
        "sf_pv_matches_solution": sf_pv_matches_solution,
        "stockfish_cp_score": 0,
    }
    return total_fitness, hero_move, details