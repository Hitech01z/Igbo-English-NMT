import ContributionForm from "../components/ContributionForm";

export default function Contribute() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 md:py-12">
      <h1 className="text-3xl md:text-4xl font-bold mb-8 md:mb-10">
        Contribute Data
      </h1>

      <ContributionForm />
    </div>
  );
}