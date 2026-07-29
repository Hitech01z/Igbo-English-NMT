import api from "./api";

export async function submitContribution(igbo, english) {

    const response = await api.post("/contribute", {
        igbo,
        english,
    });

    return response.data;
}