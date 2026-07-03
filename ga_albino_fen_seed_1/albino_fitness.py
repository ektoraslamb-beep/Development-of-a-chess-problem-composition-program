import chess
import chess.engine
import random

# ==================== MATE-IN-1 HELPER (ΜΟΝΑΔΙΚΟΤΗΤΑ) ====================

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


# ==================== ΕΥΡΕΣΗ ΥΠΟΨΗΦΙΟΥ HERO-MOVE (ALBINO) ====================

def find_candidate_albino_hero(board: chess.Board):
    """
    Αναζητά μία κίνηση Λευκού (Hero Move) που αφήνει τουλάχιστον
    ένα Λευκό πιόνι στη 2η σειρά (ο Πρωταγωνιστής P_hero).
    """
    hero_pawns = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if (
            piece
            and piece.color == chess.WHITE
            and piece.piece_type == chess.PAWN
            and chess.square_rank(sq) == 1
        ):
            hero_pawns.append(sq)

    if not hero_pawns:
        return None

    candidate_moves = []

    for m in board.legal_moves:
        piece = board.piece_at(m.from_square)

        # Αποκλεισμός βασιλιά
        if piece is not None and piece.piece_type == chess.KING:
            continue

        # Αποκλεισμός κινήσεων από τα hero pawns (θέλουμε να μείνουν στη 2η σειρά)
        if m.from_square in hero_pawns:
            continue

        b2 = board.copy()
        b2.push(m)

        # Αποκλεισμός key move που δίνει τσεκ — ο μαύρος δεν θα έχει επιλογές
        if b2.is_check():
            continue

        # Έλεγχος: πρέπει να υπάρχει τουλάχιστον ένα hero pawn στη 2η σειρά μετά την κίνηση
        current_hero_count = 0
        for sq in hero_pawns:
            piece = b2.piece_at(sq)
            if (
                piece
                and piece.color == chess.WHITE
                and piece.piece_type == chess.PAWN
                and chess.square_rank(sq) == 1
            ):
                current_hero_count += 1

        if current_hero_count > 0:
            candidate_moves.append(m)

    return candidate_moves if candidate_moves else []


# ==================== ALBINO FITNESS ====================

def evaluate_albino_fitness(board: chess.Board, engine: chess.engine.SimpleEngine):
    """
    Υπολογίζει το fitness για το θέμα Albino με ιεραρχική λογική:
    1. Ύπαρξη Hero Pawn στη 2η γραμμή.
    2. Key move από Λευκό (όχι απαραίτητα το πιόνι).
    3. Απέναντι σε κάθε άμυνα του Μαύρου, το Hero Pawn κινείται και οδηγεί σε ματ
       — είτε το ίδιο απειλεί τον βασιλιά, είτε με discovered attack.
    4. Bonus για επιπλέον κινήσεις Albino και ποιότητα κλειδιού.
    """
    score = 0.0
    penalty = 0.0
    SF_DEPTH = 10

    # 1. Βασικά φίλτρα υλικού — αυστηρότερο όριο
    if len(board.piece_map()) > 14:
        penalty -= (len(board.piece_map()) - 14) * 20.0

    # Έλεγχος υπερβολικών Βασιλισσών
    white_queens = sum(1 for p in board.piece_map().values()
                       if p.piece_type == chess.QUEEN and p.color == chess.WHITE)
    black_queens = sum(1 for p in board.piece_map().values()
                       if p.piece_type == chess.QUEEN and p.color == chess.BLACK)
    if white_queens > 1:
        penalty -= (white_queens - 1) * 300.0
    if black_queens > 1:
        penalty -= (black_queens - 1) * 300.0

    # Έλεγχος αν υπάρχει ήδη ματ σε 1 (απορρίπτεται)
    if unique_white_mate_in_1(board) is not None:
        return -1500.0, None, {"reason": "already_unique_mate_in_1"}

    # Εύρεση υποψήφιων Key Moves
    candidate_moves = find_candidate_albino_hero(board)
    if not candidate_moves:
        return -1000.0 + penalty, None, {"reason": "no_candidate_hero"}

    # Επιλογή hero_move: δοκιμάζουμε όλα τα candidates και κρατάμε
    # αυτό που είναι εξαναγκαστικό — δηλαδή μετά από αυτό ο μαύρος
    # δεν έχει καμία κίνηση που να αποφεύγει ματ σε 1 από το πιόνι
    hero_move = None
    p_hero_sq = None

    for cand in candidate_moves:
        b_test_key = board.copy()
        b_test_key.push(cand)

        # Εντοπισμός hero pawn μετά την κίνηση
        cand_hero_sq = None
        for sq in chess.SQUARES:
            piece = b_test_key.piece_at(sq)
            if (piece and piece.color == chess.WHITE and
                piece.piece_type == chess.PAWN and chess.square_rank(sq) == 1):
                cand_hero_sq = sq
                break

        if cand_hero_sq is None:
            continue

        # Έλεγχος εξαναγκασμού: για ΚΑΘΕ κίνηση του μαύρου πρέπει να υπάρχει ματ σε 1
        is_forcing = True
        for black_move in b_test_key.legal_moves:
            b_check = b_test_key.copy()
            b_check.push(black_move)
            mate = unique_white_mate_in_1(b_check)
            if mate is None:
                is_forcing = False
                break

        if is_forcing:
            hero_move = cand
            p_hero_sq = cand_hero_sq
            break

    if hero_move is None:
        return -2000.0, None, {"reason": "no_forcing_hero_move"}

    b_after_key = board.copy()
    b_after_key.push(hero_move)

    if p_hero_sq is None:
        return -1000.0 + penalty, hero_move, {"reason": "no_hero_pawn_on_2nd_rank"}

    # Bonus/penalty για θέση μαύρου βασιλιά σχετικά με το hero pawn
    bk = b_after_key.king(chess.BLACK)
    hero_file = chess.square_file(p_hero_sq)
    bk_file = chess.square_file(bk)
    bk_rank = chess.square_rank(bk)
    file_dist = abs(hero_file - bk_file)
    rank_dist = abs(1 - bk_rank)  # rank 1 είναι το hero pawn

    if file_dist <= 2 and rank_dist <= 3:
        score += 400.0
    elif file_dist > 3 or rank_dist > 4:
        score -= 300.0

    file = chess.square_file(p_hero_sq)
    rank = chess.square_rank(p_hero_sq)  # πάντα 1 (2η σειρά)

    # Τετράγωνα "στόχοι" για captures
    potential_targets = []
    if file > 0: potential_targets.append(chess.square(file - 1, rank + 1))
    if file < 7: potential_targets.append(chess.square(file + 1, rank + 1))

    for sq in potential_targets:
        target_piece = b_after_key.piece_at(sq)
        if target_piece and target_piece.color == chess.BLACK:
            score += 250.0

    # Bonus για λευκά κομμάτια στη στήλη/διαγώνιο του hero pawn
    # (υποψήφια για discovered attack με διαφορετικές κινήσεις)
    discovered_candidates = set()

    # Στήλη (για e2e3, e2e4)
    for r in range(0, rank):
        sq = chess.square(file, r)
        piece = b_after_key.piece_at(sq)
        if piece and piece.color == chess.WHITE and piece.piece_type != chess.PAWN:
            discovered_candidates.add(sq)

    # Διαγώνιες (για captures e2d3, e2f3)
    for df in [-1, 1]:
        for dr in range(1, min(rank + 1, 8)):
            f2 = file + df * dr
            r2 = rank - dr
            if 0 <= f2 <= 7 and 0 <= r2 <= 7:
                sq = chess.square(f2, r2)
                piece = b_after_key.piece_at(sq)
                if piece and piece.color == chess.WHITE and piece.piece_type != chess.PAWN:
                    discovered_candidates.add(sq)

    # Μικρό bonus — καθοδήγηση χωρίς κυριαρχία
    score += len(discovered_candidates) * 100.0

    # 2. ΕΛΕΓΧΟΣ ΜΑΤ ΜΕΤΑ ΑΠΟ ΚΑΘΕ ΚΙΝΗΣΗ ΤΟΥ ΠΙΟΝΙΟΥ (ΒΑΣΙΣΜΕΝΟ ΣΕ STOCKFISH ΑΜΥΝΕΣ)
    # Η απάντηση του Λευκού ΠΡΕΠΕΙ να είναι κίνηση του hero pawn που οδηγεί σε ματ:
    # - Άμεσα (το πιόνι απειλεί τον βασιλιά)
    # - Ή με discovered attack (το πιόνι κινείται και αποκαλύπτει άλλο κομμάτι)
    albino_mates_found = set()
    solutions = []

    try:
        # 8 καλύτερες άμυνες του μαύρου (αυξήθηκε από 5)
        defense_info = engine.analyse(b_after_key, chess.engine.Limit(depth=SF_DEPTH), multipv=8)

        for d in defense_info:
            if "pv" not in d or not d["pv"]:
                continue

            defense_move = d["pv"][0]
            b_after_defense = b_after_key.copy()
            b_after_defense.push(defense_move)

            # Το μοναδικό ματ σε 1 πρέπει να ξεκινά από το hero pawn
            # (direct ή discovered attack)
            mate_move = unique_white_mate_in_1(b_after_defense)
            if mate_move is not None and mate_move.from_square == p_hero_sq:
                albino_mates_found.add(mate_move.uci())
                solutions.append(f"Defense:{defense_move.uci()} -> Mate:{mate_move.uci()}")

    except Exception as e:
        return -2000.0, hero_move, {"reason": f"sf_error_{str(e)}"}

    # --- ΒΑΘΜΟΛΟΓΗΣΗ (ΓΡΑΜΜΙΚΗ ΚΛΙΜΑΚΑ) ---
    num_distinct_pawn_moves = len(albino_mates_found)

    if num_distinct_pawn_moves == 0:
        # Αποτυχία: Το πιόνι δεν κάνει κανένα ματ απέναντι στις σωστές άμυνες
        return -800.0 + penalty, hero_move, {"reason": "no_thematic_mate_found"}

    # Scoring με έμφαση στο 2ο variation
    # 1 variation → 300, 2 → 2000, 3 → 3500, 4 → 5000
    if num_distinct_pawn_moves == 1:
        score += 300.0
    elif num_distinct_pawn_moves == 2:
        score += 2000.0
    elif num_distinct_pawn_moves == 3:
        score += 3500.0
    elif num_distinct_pawn_moves >= 4:
        score += 5000.0

    # 3. ΠΟΙΟΤΙΚΟΣ ΕΛΕΓΧΟΣ STOCKFISH (DUALS & MATE DISTANCE)
    try:
        info_list = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=2)

        sc = info_list[0]["score"].pov(chess.WHITE)
        mate_dist = sc.mate()

        # Bonus για mate-in-2
        if mate_dist == 2:
            score += 500.0
        elif mate_dist == 1:
            return -2000.0, None, {"reason": "mate_in_1_detected"}
        elif mate_dist is not None and mate_dist > 2:
            return -1500.0, hero_move, {"reason": "mate_longer_than_2"}
        elif mate_dist is None:
            return -1500.0, hero_move, {"reason": "no_forced_mate"}

        # Ποινή για Duals (αν υπάρχει και άλλη κίνηση που δίνει ματ σε 2)
        if len(info_list) > 1:
            second_mate = info_list[1]["score"].pov(chess.WHITE).mate()
            if second_mate is not None and second_mate <= 2:
                score -= 1200.0

    except:
        pass

    score += penalty
    details = {
        "hero_uci": hero_move.uci(),
        "mates_count": num_distinct_pawn_moves,
        "solutions": solutions
    }

    return score, hero_move, details