export default function DatasetTable({
  rows
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="p-4 text-left">
              Igbo
            </th>

            <th className="p-4 text-left">
              English
            </th>
          </tr>
        </thead>

        <tbody>
          {rows.map((row, index) => (
            <tr
              key={index}
              className="border-b border-slate-800"
            >
              <td className="p-4">
                {row.igbo}
              </td>

              <td className="p-4">
                {row.english}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}