import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, ScrollArea } from "./components/ui/Card";
import { Activity, Heart, Thermometer, TrendingUp, Users } from 'lucide-react';

export default function App() {
  const rankingData = [
    { position: 1, name: "Xxxxx0", score: 95 },
    { position: 2, name: "Xxxxx1", score: 92 },
    { position: 3, name: "Xxxxx2", score: 88 },
    { position: 4, name: "Xxxxx3", score: 85 },
    { position: 5, name: "Xxxxx4", score: 82 },
  ];

  const getMedal = (position) => {
    switch (position) {
      case 1:
        return "🥇"; // Medalla de oro
      case 2:
        return "🥈"; // Medalla de plata
      case 3:
        return "🥉"; // Medalla de bronce
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-green-50 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <h1 className="text-3xl font-bold text-center text-blue-800">Panel Médico</h1>
        
        <Card className="w-full">
          <CardHeader>
            <CardTitle className="text-2xl font-semibold text-center text-blue-700">Ranking de Médicos</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px] w-full pr-4">
              {rankingData.map((item) => (
                <div key={item.position} className="flex items-center justify-between py-4 border-b border-blue-100 last:border-b-0">
                  <div className="flex items-center space-x-4">
                    <span className="text-2xl font-bold text-blue-600">
                      {getMedal(item.position)} {item.position}
                    </span>
                    <span className="text-lg text-blue-800">{item.name}</span>
                  </div>
                  <span className="text-lg font-semibold text-green-600">{item.score}</span>
                </div>
              ))}
            </ScrollArea>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-blue-700">Estadísticas Generales</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Users className="w-5 h-5 text-blue-600" />
                  <span className="text-sm text-blue-800">Pacientes Atendidos</span>
                </div>
                <span className="text-lg font-semibold text-blue-600">1,234</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-green-600" />
                  <span className="text-sm text-blue-800">Tasa de Recuperación</span>
                </div>
                <span className="text-lg font-semibold text-green-600">92%</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-blue-700">Tendencias de Salud</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-2">
                <Heart className="w-5 h-5 text-red-600" />
                <span className="text-sm text-blue-800">Puntuación de lavado de manos</span>
              </div>
              <div className="h-16 flex items-end space-x-1">
                {[60, 65, 63, 68, 70, 72, 69, 75, 72, 78].map((value, index) => (
                  <div
                    key={index}
                    className="w-6 bg-blue-400 rounded-t"
                    style={{ height: `${value}%` }}
                  ></div>
                ))}
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-600">Semana Anterior</span>
                <span className="text-xs text-blue-600">Semana Actual</span>
              </div>
              <div className="flex items-center justify-between pt-4">
                <span className="text-sm text-blue-800">Tendencia</span>
                <div className="flex items-center text-green-600">
                  <TrendingUp className="w-4 h-4 mr-1" />
                  <span className="text-sm font-semibold">+5%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
