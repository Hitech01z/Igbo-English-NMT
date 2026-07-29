import { useEffect, useState } from "react";
import api from "../services/api";

export default function DatasetExplorer() {

    const [rows, setRows] = useState([]);

    useEffect(() => {

        api.get("/dataset")
           .then((res) => setRows(res.data));

    }, []);

    return (

        <div className="p-8">

            <h1 className="text-4xl font-bold mb-8">

                Dataset Explorer

            </h1>

            <div className="overflow-auto rounded-xl">

                <table className="w-full">

                    <thead>

                        <tr className="bg-slate-800">

                            <th className="p-3 text-left">

                                Igbo

                            </th>

                            <th className="p-3 text-left">

                                English

                            </th>

                        </tr>

                    </thead>

                    <tbody>

                        {rows.map((row, i) => (

                            <tr
                                key={i}
                                className="border-b border-slate-700"
                            >

                                <td className="p-3">

                                    {row.igbo}

                                </td>

                                <td className="p-3">

                                    {row.english}

                                </td>

                            </tr>

                        ))}

                    </tbody>

                </table>

            </div>

        </div>

    );

}