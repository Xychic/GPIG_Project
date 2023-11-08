from dataclasses import dataclass

@dataclass
class SensorData:
    """DataClass representing Senosr Id in the database"""

    #TODO 
    #Decide on date format
    id:int
    date:float

    co2_level:float
    ozone_level:float
    temperature:float
    humidity:float
    co_level:float
    so2_level:float
    no2_level:float
    soil_moisture_level:float
    soil_temperature_level:float
    soil_humidity_level:float
    soil_ph:float
    anomalous:bool =False

    def __post_init__(self):
        """Run data sanitation checks for individual fields"""
        self._phValidate()
        #TODO
        #ADD more field checks if needed


    def _phValidate(self):
        if (self.soil_ph<0) or (self.soil_ph>14):
            raise ValueError(F"Invalid Ph{self.soil_ph}. 0<=ph<=14")
