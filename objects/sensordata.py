from dataclasses import dataclass


@dataclass
class SensorData:
    """DataClass representing sensor ID in the database"""

    # data should be string with format 'YYYY-MM-DD HH:MM:SS.XXXXXX'
    date: str

    co2_level: float
    ozone_level: float
    temperature: float
    humidity: float
    co_level: float
    so2_level: float
    no2_level: float
    soil_moisture_level: float
    soil_temperature_level: float
    soil_humidity_level: float
    soil_ph: float
    anomalous: bool = False

    id: int = None

    def __post_init__(self):
        """Run data sanitation checks for individual fields"""
        self._ph_validate()
        # TODO add more field checks if needed

    def to_tuple(self) -> tuple:
        """
        Convert data to tuple for easy database insertion
        :return: tuple of data in object
        """

        data = (self.co2_level, self.ozone_level, self.temperature, self.humidity, self.co_level, self.so2_level, self.no2_level,
                self.soil_moisture_level, self.soil_temperature_level, self.soil_humidity_level, self.soil_ph,
                str(self.date), self.anomalous)

        return data

    def _ph_validate(self):

        if (self.soil_ph < 0) or (self.soil_ph > 14):

            raise ValueError(F"Invalid Ph{self.soil_ph}. 0<=ph<=14")
