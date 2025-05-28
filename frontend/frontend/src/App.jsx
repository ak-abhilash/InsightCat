import React, { useState } from "react";
import axios from "axios";
import {
  UploadCloud,
  Loader2,
  TrendingUp,
  Lightbulb,
  Target,
  X,
  Download,
  CheckCircle,
  AlertTriangle,
  AlertCircle,
  XCircle,
  Database,
  FileText,
  Copy,
  Trash2,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const App = () => {
  const [file, setFile] = useState(null);
  const [insights, setInsights] = useState("");
  const [charts, setCharts] = useState([]);
  const [dataQuality, setDataQuality] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedChart, setSelectedChart] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setInsights("");
    setCharts([]);
    setDataQuality(null);
    setFileInfo(null);
    setSelectedChart(null);
  };

  const downloadChart = () => {
    if (!selectedChart) return;

    const link = document.createElement("a");
    link.href = `data:image/png;base64,${selectedChart.image}`;
    link.download = `${selectedChart.title
      .replace(/[^a-z0-9]/gi, "_")
      .toLowerCase()}.png`;
    link.click();
  };

  const closeModal = () => {
    setSelectedChart(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("https://insightcat-backend.onrender.com/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setInsights(response.data.insights);
      setCharts(response.data.charts || []);
      setDataQuality(response.data.data_quality);
      setFileInfo(response.data.file_info);
    } catch (error) {
      console.error(error);
      setInsights("Error fetching insights.");
      setCharts([]);
      setDataQuality(null);
      setFileInfo(null);
    } finally {
      setLoading(false);
    }
  };

  const getQualityStatusConfig = (status) => {
    switch (status) {
      case 'excellent':
        return { 
          icon: CheckCircle, 
          color: 'text-green-400', 
          bgColor: 'bg-green-500/20',
          borderColor: 'border-green-500/30',
          label: 'Excellent Quality'
        };
      case 'good':
        return { 
          icon: CheckCircle, 
          color: 'text-blue-400', 
          bgColor: 'bg-blue-500/20',
          borderColor: 'border-blue-500/30',
          label: 'Good Quality'
        };
      case 'needs_attention':
        return { 
          icon: AlertTriangle, 
          color: 'text-yellow-400', 
          bgColor: 'bg-yellow-500/20',
          borderColor: 'border-yellow-500/30',
          label: 'Needs Attention'
        };
      case 'poor':
        return { 
          icon: XCircle, 
          color: 'text-red-400', 
          bgColor: 'bg-red-500/20',
          borderColor: 'border-red-500/30',
          label: 'Poor Quality'
        };
      default:
        return { 
          icon: AlertCircle, 
          color: 'text-gray-400', 
          bgColor: 'bg-gray-500/20',
          borderColor: 'border-gray-500/30',
          label: 'Unknown Quality'
        };
    }
  };

  const renderDataQuality = () => {
    if (!dataQuality) return null;

    const statusConfig = getQualityStatusConfig(dataQuality.status);
    const StatusIcon = statusConfig.icon;

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
            <Database className="w-4 h-4 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-white">Data Overview</h2>
        </div>
        
        <div className="grid gap-6 md:grid-cols-2">
          {/* File Information */}
          <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white">File Information</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Filename:</span>
                <span className="text-white font-medium">{fileInfo?.filename || 'Unknown'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">File Type:</span>
                <span className="text-white font-medium">{fileInfo?.file_type || 'Unknown'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Rows:</span>
                <span className="text-white font-medium">{dataQuality.total_rows?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Columns:</span>
                <span className="text-white font-medium">{dataQuality.total_columns}</span>
              </div>
            </div>
          </div>

          {/* Data Quality Score */}
          <div className={`bg-slate-900/80 backdrop-blur-sm border ${statusConfig.borderColor} rounded-xl p-6`}>
            <div className="flex items-center gap-3 mb-4">
              <div className={`w-10 h-10 ${statusConfig.bgColor} rounded-lg flex items-center justify-center`}>
                <StatusIcon className={`w-5 h-5 ${statusConfig.color}`} />
              </div>
              <h3 className="text-lg font-semibold text-white">Quality Assessment</h3>
            </div>
            
            <div className="space-y-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">{dataQuality.quality_score}%</div>
                <div className={`text-sm font-medium ${statusConfig.color}`}>{statusConfig.label}</div>
              </div>
              
              <div className="w-full bg-slate-700/50 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${
                    dataQuality.quality_score >= 90 ? 'bg-green-500' :
                    dataQuality.quality_score >= 70 ? 'bg-blue-500' :
                    dataQuality.quality_score >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${dataQuality.quality_score}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Data Issues */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Missing Values */}
          <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white">Missing Values</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Missing:</span>
                <span className="text-white font-medium">{dataQuality.missing_values?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Percentage:</span>
                <span className={`font-medium ${
                  dataQuality.missing_percentage > 20 ? 'text-red-400' :
                  dataQuality.missing_percentage > 10 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {dataQuality.missing_percentage}%
                </span>
              </div>
              
              {Object.keys(dataQuality.columns_with_missing || {}).length > 0 && (
                <div className="mt-4">
                  <p className="text-slate-400 text-sm mb-2">Affected Columns:</p>
                  <div className="max-h-24 overflow-y-auto space-y-1">
                    {Object.entries(dataQuality.columns_with_missing).slice(0, 5).map(([col, count]) => (
                      <div key={col} className="flex justify-between text-sm">
                        <span className="text-slate-300 truncate">{col}</span>
                        <span className="text-slate-400">{count}</span>
                      </div>
                    ))}
                    {Object.keys(dataQuality.columns_with_missing).length > 5 && (
                      <div className="text-xs text-slate-500">
                        +{Object.keys(dataQuality.columns_with_missing).length - 5} more columns
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Duplicate Rows */}
          <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                <Copy className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white">Duplicate Rows</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Duplicates:</span>
                <span className="text-white font-medium">{dataQuality.duplicate_rows?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Percentage:</span>
                <span className={`font-medium ${
                  dataQuality.duplicate_percentage > 10 ? 'text-red-400' :
                  dataQuality.duplicate_percentage > 5 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {dataQuality.duplicate_percentage}%
                </span>
              </div>
              
              <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                <div className="flex items-center gap-2 text-sm">
                  {dataQuality.duplicate_percentage === 0 ? (
                    <>
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-green-400">No duplicates found</span>
                    </>
                  ) : dataQuality.duplicate_percentage < 5 ? (
                    <>
                      <AlertTriangle className="w-4 h-4 text-yellow-400" />
                      <span className="text-yellow-400">Low duplication</span>
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-4 h-4 text-red-400" />
                      <span className="text-red-400">High duplication</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderInsights = () => {
    if (!insights || insights === "Error fetching insights.") {
      return (
        <div className="text-red-400 bg-red-950/20 border border-red-800/30 rounded-lg p-4 text-center">
          {insights || "No insights available"}
        </div>
      );
    }

    let blocks = [];

    if (Array.isArray(insights)) {
      blocks = insights;
    } else {
      blocks = insights.split(/\ud83d\udcca/).filter(Boolean); // 📊 emoji unicode
    }

    return (
      <div className="grid gap-4">
        {blocks.map((block, index) => {
          if (!block || typeof block !== "string") return null;

          const lines = block.trim().split("\n").filter((line) => line.trim());
          if (lines.length === 0) return null;

          const cleanBlock = block.trim();
          if (!cleanBlock || cleanBlock.length < 10) return null;

          let title = lines[0];
          title = title.replace(/^(\ud83d\udcca|\ud83d\udd0d|\ud83d\udca1|\ud83d\udcc8|\ud83d\udcc9|\ud83e\udd14)\s*/, "").trim();

          const points = lines.slice(1);
          let whyItMatters = "";
          let suggestedAction = "";

          points.forEach((line) => {
            const cleanLine = line.replace(/^[-\u2022]\s*/, "").trim();
            if (cleanLine.toLowerCase().includes("why it matters:")) {
              whyItMatters = cleanLine.replace(/why it matters:\s*/i, "");
            } else if (cleanLine.toLowerCase().includes("suggested action:")) {
              suggestedAction = cleanLine.replace(/suggested action:\s*/i, "");
            }
          });

          return (
            <div key={index} className="group relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-slate-600/50 transition-all duration-300">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <TrendingUp className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1 space-y-4">
                    <h3 className="text-lg font-semibold text-white leading-snug">{title}</h3>

                    {whyItMatters && (
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-amber-500/20 rounded-full flex items-center justify-center mt-0.5">
                          <Lightbulb className="w-3 h-3 text-amber-400" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-amber-300 mb-1">Why it matters</p>
                          <p className="text-sm text-slate-300 leading-relaxed">{whyItMatters}</p>
                        </div>
                      </div>
                    )}

                    {suggestedAction && (
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-green-500/20 rounded-full flex items-center justify-center mt-0.5">
                          <Target className="w-3 h-3 text-green-400" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-green-300 mb-1">Suggested action</p>
                          <p className="text-sm text-slate-300 leading-relaxed">{suggestedAction}</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="absolute inset-0 opacity-20">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23334155' fill-opacity='0.1'%3E%3Ccircle cx='7' cy='7' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
            backgroundRepeat: "repeat",
          }}
        ></div>
      </div>

      <div className="relative z-10 min-h-screen p-6">
        <div className="max-w-5xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4 py-8">
            <div className="inline-flex items-center gap-3 bg-slate-900/50 backdrop-blur-sm border border-slate-700/50 rounded-full px-6 py-3">
              <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg">
                <TrendingUp className="w-4 h-4 text-white" />
              </div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-pink-400 bg-clip-text text-transparent">
                InsightCat
              </h1>
            </div>
            <p className="text-slate-400 text-lg max-w-2xl mx-auto">
              Transform your data into actionable insights with AI-powered analysis and beautiful visualizations
            </p>
          </div>

          {/* Upload Section */}
          <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-700/50 shadow-2xl">
            <CardContent className="p-8">
              <div className="space-y-6">
                <div className="text-center">
                  <h2 className="text-xl font-semibold text-white mb-2">Upload Your Dataset</h2>
                  <p className="text-slate-400 text-sm">Upload a CSV, Excel, or JSON file to get started with your data analysis</p>
                </div>

                <div className="space-y-4">
                  <div className="relative">
                    <Input
                      type="file"
                      accept=".csv,.xlsx,.xls,.json"
                      onChange={handleFileChange}
                      className="cursor-pointer bg-slate-800/50 text-white border-slate-600/50 focus:border-blue-500/50 focus:ring-blue-500/20 file:bg-slate-700 file:text-slate-300 file:border-0 file:rounded-md file:px-4 file:py-2 file:mr-4 hover:bg-slate-700/50 transition-colors"
                    />
                  </div>

                  <Button
                    onClick={handleUpload}
                    disabled={loading || !file}
                    className="group relative bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-slate-700 disabled:to-slate-700 text-white font-medium px-8 py-2.5 rounded-lg transition-all duration-300 transform hover:scale-105 disabled:hover:scale-100 shadow-lg hover:shadow-xl disabled:shadow-none"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-lg blur opacity-0 group-hover:opacity-30 transition-opacity duration-300"></div>
                    <div className="relative flex items-center justify-center">
                      {loading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <UploadCloud className="mr-2 h-4 w-4" />
                          Generate Insights
                        </>
                      )}
                    </div>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Data Quality Section */}
          {dataQuality && renderDataQuality()}

          {/* Insights */}
          {insights && (
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-br from-amber-500 to-orange-500 rounded-lg flex items-center justify-center">
                  <Lightbulb className="w-4 h-4 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">Key Insights</h2>
              </div>
              {renderInsights()}
            </div>
          )}

          {/* Charts */}
          {charts.length > 0 && (
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-teal-500 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-4 h-4 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">Data Visualizations</h2>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                {charts.map((chart, idx) => (
                  <div
                    key={idx}
                    className="group bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 hover:border-slate-600/50 transition-all duration-300 overflow-hidden cursor-pointer rounded-lg"
                    onClick={() => setSelectedChart(chart)}
                  >
                    <div className="p-6 space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-white group-hover:text-blue-300 transition-colors">
                          {chart.title}
                        </h3>
                        <div className="text-slate-400 text-sm opacity-0 group-hover:opacity-100 transition-opacity">
                          Click to expand
                        </div>
                      </div>
                      <div className="relative overflow-hidden rounded-lg bg-white">
                        <img
                          src={`data:image/png;base64,${chart.image}`}
                          alt={chart.title}
                          className="w-full h-auto transition-transform duration-300 group-hover:scale-105"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Modal for expanded chart */}
      {selectedChart && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm">
          <div className="relative bg-slate-900 border border-slate-700 rounded-xl max-w-4xl max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between p-6 border-b border-slate-700">
              <h3 className="text-xl font-semibold text-white">{selectedChart.title}</h3>
              <div className="flex items-center gap-2">
                <Button
                  onClick={downloadChart}
                  className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download
                </Button>
                <Button
                  onClick={closeModal}
                  className="bg-slate-700 hover:bg-slate-600 text-white p-2 rounded-lg"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
            <div className="p-6">
              <div className="bg-white rounded-lg overflow-hidden">
                <img
                  src={`data:image/png;base64,${selectedChart.image}`}
                  alt={selectedChart.title}
                  className="w-full h-auto"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;