import React from "react";

export function Card({ children, className = "" }) {
  return (
    <div className={`rounded-2xl bg-zinc-900 shadow-xl p-6 ${className}`}>
      {children}
    </div>
  );
}

export function CardContent({ children }) {
  return <div className="text-zinc-100 text-base">{children}</div>;
}
