from pydantic import BaseModel, Field, field_validator


class Coordinates(BaseModel):
    """A model representing geographical coordinates."""
    lat: float = Field(ge=-90, le=90, description="Latitude of the location")
    lon: float = Field(ge=-180, le=180, description="Longitude of the location")

    @field_validator("lat", "lon")
    @classmethod
    def round_to_four_places(cls, v: float) -> float:
        return round(v, 4)

