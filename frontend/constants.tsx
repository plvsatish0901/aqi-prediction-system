
import { AQICategory } from './types';

export const COLORS = {
  primaryBlue: '#1976D2',
  freshGreen: '#43A047',
  yellow: '#FBC02D',
  orange: '#FB8C00',
  red: '#E53935',
  darkRed: '#B71C1C',
  softBG: '#F1F6FB',
};

export const getAQICategory = (aqi: number): AQICategory => {
  if (aqi <= 50) return { label: 'Good', color: COLORS.freshGreen, bgColor: 'bg-green-100', textColor: 'text-green-700', glowClass: 'glow-green' };
  if (aqi <= 100) return { label: 'Satisfactory', color: '#8BC34A', bgColor: 'bg-lime-100', textColor: 'text-lime-700', glowClass: 'glow-green' };
  if (aqi <= 200) return { label: 'Moderate', color: COLORS.yellow, bgColor: 'bg-yellow-100', textColor: 'text-yellow-700', glowClass: 'glow-yellow' };
  if (aqi <= 300) return { label: 'Poor', color: COLORS.orange, bgColor: 'bg-orange-100', textColor: 'text-orange-700', glowClass: 'glow-orange' };
  if (aqi <= 400) return { label: 'Very Poor', color: COLORS.red, bgColor: 'bg-red-100', textColor: 'text-red-700', glowClass: 'glow-red' };
  return { label: 'Severe', color: COLORS.darkRed, bgColor: 'bg-red-200', textColor: 'text-red-900', glowClass: 'glow-darkred' };
};
