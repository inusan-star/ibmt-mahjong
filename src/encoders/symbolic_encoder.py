import mjx
from mjx import utils
import numpy as np

from src.encoders.base import BaseObsEncoder


class SymbolicObsEncoder(BaseObsEncoder):
    """Symbolic encoder."""

    def __init__(self):
        self._dim = 350

    @property
    def name(self) -> str:
        return "symbolic"

    @property
    def shape(self) -> tuple:
        return (34, self._dim)

    def encode(self, obs: mjx.Observation) -> np.ndarray:
        feature = np.zeros(self.shape, dtype=np.bool_)

        # Hand
        feature[:, 0:5] = self._encode_hand(obs)

        # Draw
        feature[:, 5:7] = self._encode_draw(obs)

        # Melds
        feature[:, 7:75] = self._encode_melds(obs)

        # Discards
        feature[:, 75:195] = self._encode_discards(obs)

        # Doras
        feature[:, 195:199] = self._encode_doras(obs)

        # Self Winds
        feature[:, 199:215] = self._encode_self_winds(obs)

        # Riichis
        feature[:, 215:219] = self._encode_riichis(obs)

        # Rankings
        feature[:, 219:235] = self._encode_rankings(obs)

        # Points
        feature[:, 235:315] = self._encode_points(obs)

        # Round Wind
        feature[:, 315:319] = self._encode_round_wind(obs)

        # Round
        feature[:, 319:323] = self._encode_round(obs)

        # Honba
        feature[:, 323:327] = self._encode_honba(obs)

        # Kyotaku
        feature[:, 327:331] = self._encode_kyotaku(obs)

        # Turn
        feature[:, 331:350] = self._encode_turn(obs)

        return feature

    def _encode_hand(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 5), dtype=np.bool_)
        hand = obs.curr_hand()

        for tile_type, count in enumerate(hand.closed_tile_types()):
            for i in range(count):
                if i < 4:
                    res[tile_type, i] = True

        for tile in hand.closed_tiles():
            if tile.is_red():
                res[tile.type(), 4] = True

        return res

    def _encode_draw(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 2), dtype=np.bool_)
        events = obs.events()

        if len(events) == 0:
            return res

        last_event = events[-1]

        if last_event.type() == mjx.EventType.DRAW:
            tile = obs.draws()[-1]
            res[tile.type(), 0] = True

            if tile.is_red():
                res[tile.type(), 1] = True

        return res

    def _encode_melds(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 68), dtype=np.bool_)
        opened = [[] for _ in range(4)]
        player_red_tiles = [set() for _ in range(4)]

        for event in obs.events():
            player = event.who()
            event_type = event.type()

            if event_type in [
                mjx.EventType.CHI,
                mjx.EventType.PON,
                mjx.EventType.CLOSED_KAN,
                mjx.EventType.OPEN_KAN,
            ]:
                if len(opened[player]) >= 4:
                    continue

                tiles_count = [0] * 34

                for tile in event.open().tiles():
                    tiles_count[tile.type()] += 1

                    if tile.is_red():
                        player_red_tiles[player].add(tile.type())

                opened[player].append(tiles_count)

            elif event_type == mjx.EventType.ADDED_KAN:
                tile = event.open().last_tile()

                for tiles_count in opened[player]:
                    if tiles_count[tile.type()] == 3:
                        tiles_count[tile.type()] += 1

                        if tile.is_red():
                            player_red_tiles[player].add(tile.type())

                        break

        for player in range(4):
            relative_player = (player - obs.who() + 4) % 4
            player_offset = relative_player * 17

            for meld_idx, tiles_count in enumerate(opened[player][:4]):
                meld_tile_offset = player_offset + meld_idx * 4
                tile_pos = 0

                for tile_type, count in enumerate(tiles_count):
                    for _ in range(count):
                        if tile_pos < 4:
                            res[tile_type, meld_tile_offset + tile_pos] = True
                            tile_pos += 1

            for red_tile_type in player_red_tiles[player]:
                res[red_tile_type, player_offset + 16] = True

        return res

    def _encode_discards(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 120), dtype=np.bool_)
        discard_history = [[] for _ in range(4)]
        player_red_discards = [set() for _ in range(4)]
        player_tsumogiris = [set() for _ in range(4)]
        player_riichi_discards = [set() for _ in range(4)]
        is_riichi = [False] * 4

        for event in obs.events():
            player = event.who()
            event_type = event.type()

            if event_type == mjx.EventType.RIICHI:
                is_riichi[player] = True

            elif event_type in [mjx.EventType.DISCARD, mjx.EventType.TSUMOGIRI]:
                tile = event.tile()
                tile_type = tile.type()
                discard_history[player].append(tile_type)

                if tile.is_red():
                    player_red_discards[player].add(tile_type)

                if event_type == mjx.EventType.TSUMOGIRI:
                    player_tsumogiris[player].add(tile_type)

                if is_riichi[player]:
                    player_riichi_discards[player].add(tile_type)

        for player in range(4):
            relative_player = (player - obs.who() + 4) % 4
            player_offset = relative_player * 30

            for discard_idx, tile_type in enumerate(discard_history[player][:27]):
                res[tile_type, player_offset + discard_idx] = True

            for tile_type in player_red_discards[player]:
                res[tile_type, player_offset + 27] = True

            for tile_type in player_tsumogiris[player]:
                res[tile_type, player_offset + 28] = True

            for tile_type in player_riichi_discards[player]:
                res[tile_type, player_offset + 29] = True

        return res

    def _encode_doras(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 4), dtype=np.bool_)

        for i, dora_tile_type in enumerate(obs.doras()):
            if i < 4:
                res[dora_tile_type, i] = True

        return res

    def _encode_self_winds(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 16), dtype=np.bool_)

        for player in range(4):
            relative_player = (player - obs.who() + 4) % 4
            relative_player_offset = relative_player * 4
            res[:, relative_player_offset + ((obs.round() + player) % 4)] = True

        return res

    def _encode_riichis(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 4), dtype=np.bool_)

        for event in obs.events():
            if event.type() == mjx.EventType.RIICHI:
                relative_player = (event.who() - obs.who() + 4) % 4
                res[:, relative_player] = True

        return res

    def _encode_rankings(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 16), dtype=np.bool_)

        for player, ranking in enumerate(utils.rankings(obs.tens())):
            relative_player = (player - obs.who() + 4) % 4
            relative_player_offset = relative_player * 4
            res[:, relative_player_offset + ranking] = True

        return res

    def _encode_points(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 80), dtype=np.bool_)

        for player, point in enumerate(obs.tens()):
            relative_player = (player - obs.who() + 4) % 4
            relative_player_offset = relative_player * 20
            res[:, relative_player_offset + (max(0, min(point, 59999)) // 3000)] = True

        return res

    def _encode_round_wind(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 4), dtype=np.bool_)
        res[:, obs.round() // 4] = True

        return res

    def _encode_round(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 4), dtype=np.bool_)
        res[:, obs.round() % 4] = True

        return res

    def _encode_honba(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 4), dtype=np.bool_)
        honba = obs.honba()
        res[:, min(honba, 3)] = True

        return res

    def _encode_kyotaku(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 4), dtype=np.bool_)
        kyotaku = obs.kyotaku()
        res[:, min(kyotaku, 3)] = True

        return res

    def _encode_turn(self, obs: mjx.Observation) -> np.ndarray:
        res = np.zeros((34, 19), dtype=np.bool_)
        draw_count = 0

        for event in obs.events():
            if event.type() == mjx.EventType.DRAW:
                draw_count += 1

        turn = draw_count // 4
        res[:, min(turn, 18)] = True

        return res
