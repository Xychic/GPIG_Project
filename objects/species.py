from dataclasses import dataclass


@dataclass
class Species:

    species_id: int
    species_name: str

    def __repr__(self) -> str:

        return f"id:{self.species_id}, site name :{self.species_name}"

