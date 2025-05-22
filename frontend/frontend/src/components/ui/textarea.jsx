import React from "react";

export function Textarea({ className = "", ...props }) {
  return (
    <textarea
      className={`w-full p-3 rounded-xl bg-zinc-800 text-white border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none ${className}`}
      rows={5}
      {...props}
    />
  );
}
