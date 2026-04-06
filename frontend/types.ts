
export interface AQICategory {
  label: string;
  color: string;
  bgColor: string;
  textColor: string;
  glowClass: string;
}

export interface PollutantInputs {
  pm25: number;
  pm10: number;
  no2: number;
  so2: number;
  co: number;
  o3: number;
}

export interface ForecastDataPoint {
  day: number;
  aqi: number;
  isForecast?: boolean;
}
