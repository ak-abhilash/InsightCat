import { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);

  // This function runs when user selects a file
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 text-center p-4">
      <h1 className="text-3xl font-bold text-blue-600 mb-8">
        🚀 InsightCat: Upload your data file!
      </h1>

      {/* File input */}
      <input
        type="file"
        accept=".csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/json, .xls"
        onChange={handleFileChange}
        className="mb-4"
      />

      {/* Show selected file name */}
      {file && (
        <p className="text-gray-700">
          Selected file: <strong>{file.name}</strong>
        </p>
      )}
    </div>
  );
}

export default App;