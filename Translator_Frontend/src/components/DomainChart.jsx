import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";

const data = [
  { month: "Jan", value: 120 },
  { month: "Feb", value: 240 },
  { month: "Mar", value: 420 },
  { month: "Apr", value: 700 },
  { month: "May", value: 922 }
];

export default function DatasetGrowthChart() {
  return (
    <ResponsiveContainer
      width="100%"
      height={300}
    >
      <LineChart data={data}>
        <XAxis dataKey="month" />
        <YAxis />
        <Tooltip />

        <Line
          type="monotone"
          dataKey="value"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}