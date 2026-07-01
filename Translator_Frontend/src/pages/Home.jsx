import Hero from "../components/Hero";
import FeatureCard from "../components/FeatureCard";

export default function Home() {
  const features = [
    {
      title: "Neural Translation",
      description: "Igbo ↔ English bidirectional translation.",
    },
    {
      title: "Dataset Contribution",
      description: "Community-powered corpus building.",
    },
    {
      title: "Analytics Dashboard",
      description: "Real-time project statistics.",
    },
  ];

  return (
    <div>
      <Hero />

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-16">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <FeatureCard key={feature.title} {...feature} />
          ))}
        </div>
      </section>
    </div>
  );
}