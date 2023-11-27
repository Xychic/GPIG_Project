from dataclasses import dataclass


@dataclass
class Plant:
    species_id: int
    is_diseased: bool
    date_recorded: str

    plant_id: int = None

    def __repr__(self) -> str:
        return (
            f"id: {self.plant_id}, species_id: {self.species_id}, is_diseased: {self.is_diseased}, "
            f"date_recorded: {self.date_recorded}"
        )

    def to_tuple(self) -> tuple:
        return self.species_id, self.is_diseased, self.date_recorded
