import React from "react";

export function Button({ children, onClick, className = "", ...props }) {
  return (
    <button
      onClick={onClick}
      className={`bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-xl transition-all duration-300 ease-in-out shadow-md ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
