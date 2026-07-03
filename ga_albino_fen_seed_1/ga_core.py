import csv
import os
import random
import copy
import chess
import chess.engine
import time

ENGINE_PATH = r"C:\Users\Ektoras\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

from albino_fitness import evaluate_albino_fitness

# ================= ΡΥΘΜΙΣΕΙΣ GA ===================

SEED_FEN = "8/2KN3p/4R2n/8/8/1QN2r2/3kP1B1/3r4 w - - 0 1"
POPULATION_SIZE = 200
GENERATIONS = 600
ELITE_COUNT = 10
MUTATIONS_PER_CHILD = (2, 6)
CANDIDATES_CSV = "albino_candidates.csv"
CANDIDATE_FITNESS_THRESHOLD = 2200
CROSSOVER_RATE = 0.6

# =============== UTILS ΓΙΑ SEEDS ===================

def load_seed_fens(path):
    fens = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            if not row:
                continue
            fen = row[0]
            try:
                board = chess.Board(fen)
                if basic_board_filters(board):
                    fens.append(fen)
            except Exception:
                continue
    return fens

# =============== BASIC FILTERS =====================

def basic_board_filters(board: chess.Board) -> bool:
    if not board.is_valid():
        return False

    if board.is_check():
        return False

    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return False

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return False

    bk_rank = chess.square_rank(bk)
    if bk_rank < 1:
        return False
    if bk_rank > 4:
        return False

    if chess.square_distance(wk, bk) <= 1:
        return False

    if len(board.piece_map()) > 16:
        return False

    # Έλεγχος υπερβολικών Βασιλισσών
    white_queens = sum(1 for p in board.piece_map().values()
                       if p.piece_type == chess.QUEEN and p.color == chess.WHITE)
    black_queens = sum(1 for p in board.piece_map().values()
                       if p.piece_type == chess.QUEEN and p.color == chess.BLACK)
    if white_queens > 1:
        return False
    if black_queens > 1:
        return False

    return True

# =============== MUTATION HELPERS ==================

def random_empty_square(board: chess.Board, max_tries=100):
    for _ in range(max_tries):
        sq = random.randrange(64)
        if board.piece_at(sq) is None:
            return sq
    return None

def random_piece_square(board: chess.Board, max_tries=100):
    pieces = list(board.piece_map().keys())
    random.shuffle(pieces)
    for sq in pieces:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        if piece.piece_type == chess.KING:
            continue
        return sq
    return None

def mutate_move_piece(board: chess.Board):
    from_sq = random_piece_square(board)
    if from_sq is None:
        return
    to_sq = random_empty_square(board)
    if to_sq is None:
        return
    piece = board.piece_at(from_sq)
    board.remove_piece_at(from_sq)
    board.set_piece_at(to_sq, piece)

def mutate_add_piece(board: chess.Board):
    to_sq = random_empty_square(board)
    if to_sq is None:
        return
    candidate_pieces = [
        chess.Piece(chess.PAWN,   chess.WHITE),
        chess.Piece(chess.PAWN,   chess.BLACK),
        chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.Piece(chess.BISHOP, chess.WHITE),
        chess.Piece(chess.BISHOP, chess.BLACK),
        chess.Piece(chess.ROOK,   chess.WHITE),
        chess.Piece(chess.ROOK,   chess.BLACK),
        chess.Piece(chess.QUEEN,  chess.WHITE),
        chess.Piece(chess.QUEEN,  chess.BLACK),
    ]
    piece = random.choice(candidate_pieces)
    board.set_piece_at(to_sq, piece)

def mutate_remove_piece(board: chess.Board):
    sq = random_piece_square(board)
    if sq is None:
        return
    piece = board.piece_at(sq)
    if piece.piece_type == chess.PAWN and piece.color == chess.WHITE and chess.square_rank(sq) == 1:
        return
    board.remove_piece_at(sq)

def mutate_swap_type(board: chess.Board):
    sq = random_piece_square(board)
    if sq is None:
        return
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type == chess.KING:
        return
    color = piece.color
    candidate_types = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    new_type = piece.piece_type
    while new_type == piece.piece_type:
        new_type = random.choice(candidate_types)
    new_piece = chess.Piece(new_type, color)
    board.set_piece_at(sq, new_piece)

def mutate_board(board: chess.Board, num_mutations: int):
    for _ in range(num_mutations):
        old_fen = board.fen()
        op = random.choice(["move", "add", "remove", "swap_type"])
        snapshot = board.copy(stack=False)

        if op == "move":
            mutate_move_piece(board)
        elif op == "add":
            mutate_add_piece(board)
        elif op == "remove":
            mutate_remove_piece(board)
        elif op == "swap_type":
            mutate_swap_type(board)

        if not basic_board_filters(board):
            board.set_fen(old_fen)

# =============== CROSSOVER =========================

def crossover_boards(board1: chess.Board, board2: chess.Board, max_tries=10) -> chess.Board:
    """
    Piece-type crossover:
    - Λευκά κομμάτια (εκτός βασιλιά) από τον board1
    - Μαύρα κομμάτια (εκτός βασιλιά) από τον board2
    - Fallback στον board1 αν δεν βγει valid θέση
    """
    for _ in range(max_tries):
        new_board = chess.Board(fen=None)
        new_board.clear()

        wk = board1.king(chess.WHITE)
        if wk is not None:
            new_board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))

        bk = board2.king(chess.BLACK)
        if bk is not None:
            new_board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        for sq, piece in board1.piece_map().items():
            if piece.color == chess.WHITE and piece.piece_type != chess.KING:
                if new_board.piece_at(sq) is None:
                    new_board.set_piece_at(sq, piece)

        for sq, piece in board2.piece_map().items():
            if piece.color == chess.BLACK and piece.piece_type != chess.KING:
                if new_board.piece_at(sq) is None:
                    new_board.set_piece_at(sq, piece)

        new_board.turn = chess.WHITE

        if basic_board_filters(new_board):
            return new_board

    return board1.copy(stack=False)

# =============== INDIVIDUAL ========================

class Individual:
    def __init__(self, board: chess.Board):
        self.board = board
        self.fitness = None
        self.hero_move = None
        self.details = None

    def evaluate(self, engine):
        score, hero_move, details = evaluate_albino_fitness(self.board, engine)
        self.fitness = score
        self.hero_move = hero_move
        self.details = details

# =============== GA CORE ===========================

def init_population(pop_size: int):
    population = []
    while len(population) < pop_size:
        b = chess.Board(SEED_FEN)
        num_mut = random.randint(1, 3)
        mutate_board(b, num_mut)
        if not basic_board_filters(b):
            continue
        ind = Individual(b)
        population.append(ind)
    return population

def tournament_select(population, k=3):
    cand = random.sample(population, k)
    cand.sort(key=lambda ind: ind.fitness, reverse=True)
    return cand[0]

def make_next_generation(population):
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    new_pop = []

    # Elitism
    elite = population[:ELITE_COUNT]
    for e in elite:
        clone_board_obj = e.board.copy(stack=False)
        clone = Individual(clone_board_obj)
        clone.fitness = e.fitness
        clone.hero_move = e.hero_move
        clone.details = e.details
        new_pop.append(clone)

    # Random immigration (15%) — από το seed FEN με mutations
    immigrants_count = int(len(population) * 0.15)
    for _ in range(immigrants_count):
        b = chess.Board(SEED_FEN)
        mutate_board(b, random.randint(1, 3))
        if basic_board_filters(b):
            new_pop.append(Individual(b))

    # Crossover + Mutation
    while len(new_pop) < len(population):
        use_crossover = random.random() < CROSSOVER_RATE

        if use_crossover:
            parent1 = tournament_select(population, k=3)
            parent2 = tournament_select(population, k=3)
            child_board = crossover_boards(parent1.board, parent2.board)
        else:
            parent1 = tournament_select(population, k=3)
            child_board = parent1.board.copy(stack=False)

        num_mut = random.randint(MUTATIONS_PER_CHILD[0], MUTATIONS_PER_CHILD[1])
        mutate_board(child_board, num_mut)

        if not basic_board_filters(child_board):
            continue

        child = Individual(child_board)
        new_pop.append(child)

    return new_pop

# =============== MAIN LOOP =========================

def main():
    overall_start_time = time.time()

    print(f"[i] Seed FEN: {SEED_FEN}")

    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        print(f"[i] Stockfish engine opened at {ENGINE_PATH}")
    except Exception as e:
        print(f"[i] Αποτυχια φορτωσης Stockfish : {e}")
        return

    population = init_population(POPULATION_SIZE)
    best_global = None

    for gen in range(GENERATIONS):
        start_time = time.time()

        for ind in population:
            if ind.fitness is None:
                ind.evaluate(engine)

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        best = population[0]

        if best_global is None or best.fitness > best_global.fitness:
            best_global = copy.deepcopy(best)

        hero_uci = best.hero_move.uci() if best.hero_move is not None else "None"
        print(f"[Gen {gen}] best fitness = {best.fitness:.2f} | hero = {hero_uci}")
        print(f"          FEN: {best.board.fen()}")

        # Κατανομή variations στον πληθυσμό
        var_0 = sum(1 for ind in population if ind.details and ind.details.get("mates_count", 0) == 0)
        var_1 = sum(1 for ind in population if ind.details and ind.details.get("mates_count", 0) == 1)
        var_2p = sum(1 for ind in population if ind.details and ind.details.get("mates_count", 0) >= 2)
        print(f"          Κατανομη: 0-var={var_0} | 1-var={var_1} | 2+-var={var_2p}")

        population = make_next_generation(population)
        gen_end_time = time.time()
        print(f"[i] Γενια {gen} τελειωσε σε {gen_end_time - start_time:.2f} δευτερολεπτα")

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)

    file_exists = os.path.isfile(CANDIDATES_CSV)

    with open(CANDIDATES_CSV, "a", newline="", encoding="utf-8") as cand_file:
        cand_writer = csv.writer(cand_file)

        if not file_exists:
            cand_writer.writerow(["fitness", "fen", "hero_uci", "sf_cp_score", "sf_top_move", "solutions"])

        if best_global:
            sol_list = best_global.details.get("solutions", [])
            solutions_display = " | ".join(sol_list)

            cand_writer.writerow([
                f"{best_global.fitness:.2f}",
                f"{minutes}m {seconds}s",
                best_global.board.fen(),
                best_global.hero_move.uci() if best_global.hero_move else "None",
                best_global.details.get("stockfish_cp_score", "N/A"),
                best_global.details.get("top_move_sf", "None"),
                solutions_display
            ])

    print("\n" + "="*50)
    print(f"[i] GA Phase 1 ολοκληρώθηκε.")
    print(f"[i] ΣΥΝΟΛΙΚΟΣ ΧΡΟΝΟΣ ΕΚΤΕΛΕΣΗΣ: {minutes} λεπτά και {seconds} δευτερόλεπτα")
    print(f"[i] Το καλύτερο FEN αποθηκεύτηκε στο {CANDIDATES_CSV}")
    print(f"[i] Καλύτερο άτομο συνολικά: fitness = {best_global.fitness:.2f}")
    print(f"[i] FEN: {best_global.board.fen()}")

    if best_global.hero_move is not None:
        print(f"[i] Hero move (uci): {best_global.hero_move.uci()}")
        print("[i] Solutions Found:")
        for sol in best_global.details.get("solutions", []):
            print(f"    - {sol}")

    print("="*50)
    engine.quit()

if __name__ == "__main__":
    main()