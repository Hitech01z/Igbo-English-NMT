import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";

const data = [
  { month: "Jan", value: 100 },
  { month: "Feb", value: 250 },
  { month: "Mar", value: 400 },
  { month: "Apr", value: 600 },
  { month: "May", value: 750 },
  { month: "Jun", value: 922 }
];

export default function DatasetGrowthChart() {
  return (
    <div className="h-80">

      <ResponsiveContainer
        width="100%"
        height="100%"
      >

        <LineChart data={data}>

          <CartesianGrid strokeDasharray="3 3" />

          <XAxis dataKey="month" />

          <YAxis />

          <Tooltip />

          <Line
            type="monotone"
            dataKey="value"
            stroke="#10b981"
          />

        </LineChart>

      </ResponsiveContainer>

    </div>
  );
}