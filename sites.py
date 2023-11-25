from dataclasses import dataclass

@dataclass
class site():
    id:int
    site_name:str

    def __repr__(self) -> str:
        return f"id:{self.id}, site name :{self.site_name}"

