from dataclasses import dataclass


@dataclass
class Site:
    site_id: int
    site_name: str

    def __repr__(self) -> str:
        return f"id: {self.site_id}, site name: {self.site_name}"
