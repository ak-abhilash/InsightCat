import React from "react";

export function Input({ type = "text", className = "", ...props }) {
  return (
    <input
      type={type}
      className={`w-full p-3 rounded-xl bg-zinc-800 text-white border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 ${className}`}
      {...props}
    />
  );
}
