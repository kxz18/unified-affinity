#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy, deepcopy
from typing import List, Iterator, Optional

from utils import const

import os


class MoleculeVocab:

    MAX_ATOM_NUMBER = 14

    def __init__(self, use_small_mol=True):
        self.PAD, self.MASK, self.UNK = '#', '*', '?' # pad / mask / unk
        specials = [# special added
                (self.PAD, 'PAD'), (self.MASK, 'MASK'), (self.UNK, 'UNK'), # pad / mask / unk
            ]
        
        aas = const.aas
        bases = const.bases

        if use_small_mol:
            sms = [(e.lower(), e) for e in const.periodic_table]
        else:
            sms = [] # disable small molecule vocabulary

        self.atom_pad, self.atom_mask = 'pad', 'msk', # Avoid conflict with atom P
        self.atom_pos_pad, self.atom_pos_mask = 'pad', 'msk'
        self.atom_pos_sm = 'sml'  # small molecule

        # block level vocab
        self.idx2block = specials + aas + bases + sms 
        self.symbol2idx, self.abrv2idx = {}, {}
        for i, (symbol, abrv) in enumerate(self.idx2block):
            self.symbol2idx[symbol] = i
            self.abrv2idx[abrv] = i
        self.special_mask = [1 for _ in specials] + [0 for _ in aas + bases + sms]

        # atom level vocab
        self.idx2atom = [self.atom_pad, self.atom_mask] + const.periodic_table
        self.idx2atom_pos = [self.atom_pos_pad, self.atom_pos_mask] + \
                            ['', 'A', 'B', 'G', 'D', 'E', 'Z', 'H', 'XT', 'P'] + \
                            [self.atom_pos_sm, "'"] # SM is for atoms in small molecule, 'P' for O1P, O2P, O3P
        self.atom2idx, self.atom_pos2idx = {}, {}
        self.atom2idx = {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i
    
    # block level APIs

    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.idx2block[idx][0]

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.idx2block[idx][1]

    def abrv_to_idx(self, abrv):
        abrv = abrv.upper()
        return self.abrv2idx.get(abrv, self.abrv2idx['UNK'])

    def symbol_to_idx(self, symbol):
        # symbol = symbol.upper()
        return self.symbol2idx.get(symbol, self.abrv2idx['UNK'])
    
    def idx_to_symbol(self, idx):
        return self.idx2block[idx][0]

    def idx_to_abrv(self, idx):
        return self.idx2block[idx][1]

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)
    
    def get_special_mask(self):
        return copy(self.special_mask)
    
    # atom level APIs 

    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]
    
    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]
    
    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        atom = atom.upper()
        return self.atom2idx.get(atom, self.atom2idx[self.atom_mask])

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]
    
    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx.get(atom_pos, self.atom_pos2idx[self.atom_pos_mask])

    # sizes

    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_block_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)

    # others
    @property
    def ca_channel_idx(self):
        return const.backbone_atoms.index('CA')

VOCAB = MoleculeVocab()


class Atom:
    def __init__(self, atom_name: str, coordinate: List[float], element: str, pos_code: str=None, properties: Optional[dict]=None):
        self.name = atom_name
        self.coordinate = coordinate
        self.element = element
        if pos_code is None:
            pos_code = atom_name.lstrip(element)
            # pos_code = ''.join((c for c in pos_code if not c.isdigit()))
            self.pos_code = pos_code
        else:
            self.pos_code = pos_code
        self.properties = {} if properties is None else deepcopy(properties)

    def get_element(self):
        return self.element
    
    def get_coord(self):
        return copy(self.coordinate)
    
    def get_pos_code(self):
        return self.pos_code
    
    def get_property(self, name: str, default_value=None):
        return self.properties.get(name, default_value)
    
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Atom ({self.name}): {self.element}({self.pos_code}) [{','.join(['{:.4f}'.format(num) for num in self.coordinate])}]"
    
    def to_tuple(self):
        return (
            self.name,
            self.coordinate,
            self.element,
            self.pos_code,
            self.properties
        )
    
    @classmethod
    def from_tuple(self, data):
        return Atom(
            atom_name=data[0],
            coordinate=data[1],
            element=data[2],
            pos_code=data[3],
            properties=data[4] if len(data) == 5 else None
        )


class Block:
    def __init__(self, abrv: str, units: List[Atom], id: Optional[any]=None, properties: Optional[dict]=None) -> None:
        self.abrv: str = abrv
        self.units: List[Atom] = units
        self._uname2idx = { unit.name: i for i, unit in enumerate(self.units) }
        self.id = id
        self.properties = {} if properties is None else deepcopy(properties)

    def __len__(self) -> int:
        return len(self.units)
    
    def __iter__(self) -> Iterator[Atom]:
        return iter(self.units)
    
    def get_unit_by_name(self, name: str) -> Atom:
        idx = self._uname2idx[name]
        return self.units[idx]
    
    def has_unit(self, name: str) -> bool:
        return name in self._uname2idx

    def get_property(self, name: str):
        return self.properties.get(name, None)

    def to_tuple(self):
        return (
            self.abrv,
            [unit.to_tuple() for unit in self.units],
            self.id,
            self.properties
        )
    
    def is_residue(self):
        return self.has_unit('CA') and self.has_unit('N') and self.has_unit('C') and self.has_unit('O')
   
    @classmethod
    def from_tuple(self, data):
        return Block(
            abrv=data[0],
            units=[Atom.from_tuple(unit_data) for unit_data in data[1]],
            id=data[2],
            properties=data[3]
        )
    
    def __repr__(self) -> str:
        return f"Block ({self.abrv}), id {self.id}:\n\t" + '\n\t'.join([repr(at) for at in self.units]) + '\n' + f'properties: {self.properties}'