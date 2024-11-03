import React from 'react';

export const Card = ({ className, children }) => (
  <div className={`bg-white shadow-md rounded-lg p-4 ${className}`}>{children}</div>
);

export const CardContent = ({ children }) => (
  <div className="p-4">{children}</div>
);

export const CardHeader = ({ children }) => (
  <div className="border-b pb-2 mb-2">{children}</div>
);

export const CardTitle = ({ children }) => (
  <h2 className="text-xl font-bold">{children}</h2>
);

export const ScrollArea = ({ children, className }) => (
  <div className={`overflow-y-auto ${className}`}>{children}</div>
);
