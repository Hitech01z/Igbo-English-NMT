import csv
from pathlib import Path


OUTPUT_FILE = Path("dataset/monolingual/igbo.csv")


sentences = [

    # AGRICULTURE
    ("Onye ọrụ ugbo na-arụsi ọrụ ike n'ubi.", "agriculture"),
    ("Ọka dị n'ubi ahụ toro nke ọma.", "agriculture"),
    ("E nwere ọtụtụ osisi n'ugbo anyị.", "agriculture"),
    ("Mmiri ozuzo nyere ihe ọkụkụ aka ito.", "agriculture"),
    ("Onye ọrụ ugbo kụrụ akwụkwọ nri n'ubi.", "agriculture"),
    ("A na-akụ ji mgbe oge ruru.", "agriculture"),
    ("Ụmụ nwoke na-enyere nna ha aka n'ubi.", "agriculture"),
    ("Ehi na-ata ahịhịa n'ala ubi.", "agriculture"),
    ("Ọrụ ugbo chọrọ ndidi na ike.", "agriculture"),
    ("A na-egbute ihe ubi mgbe ha tozuru.", "agriculture"),
    ("Ọtụtụ ndị mmadụ na-adabere n'ọrụ ugbo.", "agriculture"),
    ("Onye ọrụ ugbo zụtara mkpụrụ ọhụrụ.", "agriculture"),
    ("A na-echekwa ọka n'ebe kpọrọ nkụ.", "agriculture"),
    ("Ọkụ nwere ike imebi ihe ọkụkụ.", "agriculture"),
    ("Nne m na-akụ akwụkwọ nri n'azụ ụlọ.", "agriculture"),
    ("Ewu na ọkụkọ bụ anụ ụlọ ndị a na-ahụkarị.", "agriculture"),
    ("Ọrụ ugbo bụ isi iyi nri maka ọtụtụ ezinụlọ.", "agriculture"),
    ("A na-eji mmiri agba ihe ọkụkụ mmiri n'oge ọkọchị.", "agriculture"),
    ("Onye ọrụ ugbo gara ahịa ree ihe ubi ya.", "agriculture"),
    ("Ala ọma na-enyere ihe ọkụkụ aka ito ngwa ngwa.", "agriculture"),

    # BUSINESS
    ("Onye ahịa ahụ meghere ụlọ ahịa ya n'ụtụtụ.", "business"),
    ("Ọtụtụ ndị mmadụ na-azụ ahịa n'ahịa a.", "business"),
    ("Ọnụahịa ngwaahịa ahụ gbagoro n'izu a.", "business"),
    ("Onye ahịa chọrọ ịmata ọnụ ahịa ihe ahụ.", "business"),
    ("Ụlọ ọrụ ahụ nwere ọtụtụ ndị ọrụ.", "business"),
    ("Ha na-eme atụmatụ ịmalite azụmahịa ọhụrụ.", "business"),
    ("Onye ahịa ahụ rere ngwaahịa ya niile.", "business"),
    ("Azụmahịa ọma chọrọ atụmatụ siri ike.", "business"),
    ("Ọ na-ere uwe n'ahịa obodo.", "business"),
    ("Onye ahịa ahụ nwetara uru n'azụmahịa ya.", "business"),
    ("Ụlọ ahịa ahụ na-emeghe kwa ụbọchị.", "business"),
    ("Ọtụtụ ndị mmadụ na-eji ekwentị eme azụmahịa.", "business"),
    ("Onye ahịa ahụ nyere onye ahịa akwụkwọ nnata.", "business"),
    ("Ọ dị mkpa ịkwado ndị ahịa nke ọma.", "business"),
    ("Ụlọ ọrụ ahụ chọrọ ịgbasa azụmahịa ya.", "business"),
    ("Ọtụtụ ụlọ ọrụ na-eji teknụzụ arụ ọrụ.", "business"),
    ("Onye ahịa ahụ debere ego n'ụlọ akụ.", "business"),
    ("Ahịa ngwaahịa ahụ dị elu taa.", "business"),
    ("Ha nwere nzukọ gbasara azụmahịa ha.", "business"),
    ("Onye nwe ụlọ ahịa ahụ na-elekọta ndị ọrụ ya.", "business"),

    # CULTURE
    ("Omenala Igbo nwere ọtụtụ ihe pụrụ iche.", "culture"),
    ("Ndị obodo na-eme emume kwa afọ.", "culture"),
    ("Ezinụlọ ahụ kwadebere maka emume ọdịnala.", "culture"),
    ("Ndị okenye na-akụziri ụmụaka omenala.", "culture"),
    ("A na-asọpụrụ ndị okenye n'obodo.", "culture"),
    ("Egwu ọdịnala na-atọ ọtụtụ mmadụ ụtọ.", "culture"),
    ("Ndị mmadụ gbakọtara maka emume ahụ.", "culture"),
    ("Akụkọ ndị nna nna bara uru nke ukwuu.", "culture"),
    ("Ụmụaka na-amụta egwu ọdịnala n'ụlọ akwụkwọ.", "culture"),
    ("Omenala na-ejikọta ndị mmadụ ọnụ.", "culture"),
    ("Ndị obodo na-akwado emume ha nke ọma.", "culture"),
    ("A na-eji egwu eme emume dị iche iche.", "culture"),
    ("Ndị okenye kọọrọ ụmụaka akụkọ n'abalị.", "culture"),
    ("Asụsụ bụ akụkụ dị mkpa nke omenala.", "culture"),
    ("Ọtụtụ ezinụlọ na-echekwa omenala ha.", "culture"),
    ("Emume ahụ malitere n'ehihie.", "culture"),
    ("Ndị mmadụ yi uwe ọdịnala n'emume ahụ.", "culture"),
    ("A na-akụziri ụmụaka nkwanye ùgwù.", "culture"),
    ("Omenala na-agbanwe nwayọọ nwayọọ.", "culture"),
    ("Obodo ahụ nwere ọtụtụ akụkọ ihe mere eme.", "culture"),

    # DAILY CONVERSATION
    ("Kedu ka ị mere taa?", "daily_conversation"),
    ("A mere m nke ọma.", "daily_conversation"),
    ("Ebee ka ị na-aga?", "daily_conversation"),
    ("Ana m aga n'ụlọ.", "daily_conversation"),
    ("Biko nyere m aka.", "daily_conversation"),
    ("Achọrọ m ịjụ gị ajụjụ.", "daily_conversation"),
    ("Gịnị ka ị na-eme ugbu a?", "daily_conversation"),
    ("Ana m agụ akwụkwọ.", "daily_conversation"),
    ("Ị nwere oge taa?", "daily_conversation"),
    ("Ee, enwere m oge.", "daily_conversation"),
    ("Biko chere obere oge.", "daily_conversation"),
    ("A ga m alaghachi echi.", "daily_conversation"),
    ("Echegbula onwe gị.", "daily_conversation"),
    ("Aghọtara m ihe ị kwuru.", "daily_conversation"),
    ("Amaghị m azịza ya.", "daily_conversation"),
    ("Ka anyị gaa ọnụ.", "daily_conversation"),
    ("Ọ dị mma, enweghị nsogbu.", "daily_conversation"),
    ("Kpọọ m mgbe ị rutere.", "daily_conversation"),
    ("A hụrụ m gị n'anya.", "daily_conversation"),
    ("Ka chi fo.", "daily_conversation"),

    # EDUCATION
    ("Ụmụ akwụkwọ na-amụ ihe n'ụlọ akwụkwọ.", "education"),
    ("Onye nkuzi na-akọwa ihe ọhụrụ.", "education"),
    ("Akwụkwọ a nwere ọtụtụ ihe ọmụma.", "education"),
    ("Ọmụmụ ihe chọrọ mgbalị.", "education"),
    ("Ụmụ akwụkwọ na-akwadebe maka ule.", "education"),
    ("Onye nkuzi nyere ụmụ akwụkwọ ọrụ ụlọ.", "education"),
    ("Ụlọ akwụkwọ ahụ nwere nnukwu ọbá akwụkwọ.", "education"),
    ("Ọ na-agụ akwụkwọ kwa mgbede.", "education"),
    ("Ụmụ akwụkwọ na-ajụ ajụjụ n'oge klaasị.", "education"),
    ("Agụmakwụkwọ dị mkpa maka ọdịnihu.", "education"),
    ("Onye nkuzi dere ihe n'elu bọọdụ.", "education"),
    ("Ụmụ akwụkwọ bịara klaasị n'oge.", "education"),
    ("Akwụkwọ kọmputa dị n'elu tebụl.", "education"),
    ("Ọ na-amụ sayensị na mgbakọ na mwepụ.", "education"),
    ("Ule ahụ ga-amalite n'izu na-abịa.", "education"),
    ("Ụmụ akwụkwọ na-arụ ọrụ ọnụ.", "education"),
    ("Ọgụgụ na-enyere mmadụ aka ịmụ ihe.", "education"),
    ("Onye nkuzi jụrụ ajụjụ dị mkpa.", "education"),
    ("Ụmụ akwụkwọ ahụ gafere ule ha.", "education"),
    ("Ọ chọrọ ịga mahadum n'ọdịnihu.", "education"),

    # GENERAL
    ("Ụtụtụ a dị jụụ nke ukwuu.", "general"),
    ("Anyanwụ na-enwu taa.", "general"),
    ("E nwere ọtụtụ mmadụ n'okporo ụzọ.", "general"),
    ("Nwa ahụ na-egwu egwu n'èzí.", "general"),
    ("Nne m na-esi nri n'ọnụ ụlọ.", "general"),
    ("Anyị gara njem n'ụtụtụ.", "general"),
    ("Ụlọ ahụ dị nso n'ahịa.", "general"),
    ("Ọtụtụ mmadụ nọ n'ụlọ.", "general"),
    ("Akwụkwọ ahụ dị n'elu tebụl.", "general"),
    ("Ọ na-ehi ụra ugbu a.", "general"),
    ("Ụbọchị taa dị ọkụ.", "general"),
    ("Ndị enyi ahụ na-akparịta ụka.", "general"),
    ("Ọ zụtara uwe ọhụrụ.", "general"),
    ("Anyị na-echere ụgbọ ala.", "general"),
    ("Ọtụtụ ụmụaka na-egwu bọl.", "general"),
    ("Nne na nna ya nọ n'ụlọ.", "general"),
    ("A na-anụ egwu n'ebe dị anya.", "general"),
    ("Ọ gara ịzụ nri.", "general"),
    ("Ha biri n'obodo ahụ ogologo oge.", "general"),
    ("Anyị ga-ahụ ibe anyị echi.", "general"),

    # GOVERNMENT
    ("Gọọmenti na-arụ ọrụ maka ụmụ amaala.", "government"),
    ("Ndị obodo chọrọ ezi ọchịchị.", "government"),
    ("A họpụtara onye isi ọhụrụ.", "government"),
    ("Ụlọ ọrụ gọọmenti na-enye ọrụ dị iche iche.", "government"),
    ("Ndị mmadụ gara votu n'ụbọchị ntuli aka.", "government"),
    ("Iwu obodo kwesịrị ichebe ụmụ amaala.", "government"),
    ("Gọọmenti kwupụtara atụmatụ ọhụrụ.", "government"),
    ("Ndị isi obodo nwere nzukọ.", "government"),
    ("Ụlọ ọgwụ ọha na-enyere ọtụtụ mmadụ aka.", "government"),
    ("Ọrụ gọọmenti chọrọ ndị ọrụ nwere ahụmahụ.", "government"),
    ("Ndị mmadụ na-atụ anya mgbanwe dị mma.", "government"),
    ("Onye isi ahụ kwuru okwu n'ihu ọha.", "government"),
    ("A na-eme nzukọ maka mmepe obodo.", "government"),
    ("Obodo chọrọ ezigbo okporo ụzọ.", "government"),
    ("Ndị ọrụ na-arụ ọrụ n'ụlọ ọrụ ọha.", "government"),
    ("Iwu ọhụrụ malitere ịrụ ọrụ.", "government"),
    ("Ndị obodo kwuru echiche ha.", "government"),
    ("Gọọmenti na-akwado agụmakwụkwọ.", "government"),
    ("Ndị ọchịchị kwesịrị ige ntị n'ọnụ ndị mmadụ.", "government"),
    ("Mmepe obodo chọrọ nkwado onye ọ bụla.", "government"),

    # HEALTH
    ("Ahụ ike dị mkpa maka mmadụ niile.", "health"),
    ("Ọ gara ụlọ ọgwụ n'ụtụtụ.", "health"),
    ("Dọkịta nyochara onye ọrịa ahụ.", "health"),
    ("Ọ na-enwe ahụ ọkụ.", "health"),
    ("Ndị mmadụ kwesịrị ịṅụ mmiri zuru oke.", "health"),
    ("Ịsa aka na-enyere aka igbochi ọrịa.", "health"),
    ("Ọgwụ ahụ dị n'elu tebụl.", "health"),
    ("Onye ọrịa ahụ na-agbake nke ọma.", "health"),
    ("Dọkịta nyere ya ndụmọdụ.", "health"),
    ("Ezinụlọ ahụ gara ụlọ ọgwụ.", "health"),
    ("Ọ dị mkpa iri nri na-edozi ahụ.", "health"),
    ("Ụra zuru oke na-enyere ahụ aka.", "health"),
    ("Ọ na-enwe mgbu n'afọ.", "health"),
    ("Nọọsụ nyere onye ọrịa aka.", "health"),
    ("A ga-eme nyocha ahụ echi.", "health"),
    ("Ọrịa nwere ike imetụta ọtụtụ mmadụ.", "health"),
    ("Ọ na-agbaso ndụmọdụ dọkịta.", "health"),
    ("Ahụ ya adịghị ya mma taa.", "health"),
    ("A na-enye ụmụaka ọgwụ mgbochi.", "health"),
    ("Ịdị ọcha dị mkpa maka ahụ ike.", "health"),

    # RELIGION
    ("Ndị mmadụ gara ụlọ ekpere n'ụtụtụ.", "religion"),
    ("Ọ na-ekpe ekpere kwa ụbọchị.", "religion"),
    ("Ezinụlọ ahụ gara ụka ọnụ.", "religion"),
    ("Ndị mmadụ na-ekele Chineke maka ndụ.", "religion"),
    ("Onye ụkọchukwu kwuru okwu n'ihu ọha.", "religion"),
    ("Ekpere na-enye ọtụtụ mmadụ olileanya.", "religion"),
    ("Ndị kwere ekwe gbakọtara maka emume.", "religion"),
    ("Ọ na-agụ akwụkwọ nsọ n'ụlọ.", "religion"),
    ("Ụlọ ekpere ahụ jupụtara na ndị mmadụ.", "religion"),
    ("Ha na-ekpe ekpere maka ezinụlọ ha.", "religion"),
    ("Ndị mmadụ na-enyere ndị nọ na mkpa aka.", "religion"),
    ("Okwukwe bụ ihe dị mkpa nye ọtụtụ mmadụ.", "religion"),
    ("Ezinụlọ ahụ na-ekpe ekpere ọnụ.", "religion"),
    ("A na-eme emume okpukpe kwa afọ.", "religion"),
    ("Ọ gara n'ụlọ ekpere n'ehihie.", "religion"),
    ("Ndị mmadụ na-ege ntị n'okwu onye ndu.", "religion"),
    ("Ọtụtụ mmadụ na-achọ udo na olileanya.", "religion"),
    ("Ha nyere ndị ogbenye nri.", "religion"),
    ("Ndị mmadụ gbakọtara n'udo.", "religion"),
    ("Ọ na-akwanyere nkwenkwe ndị ọzọ ùgwù.", "religion"),

    # TECHNOLOGY
    ("Kọmputa dị mkpa n'oge a.", "technology"),
    ("Ọ na-eji ekwentị ya kwa ụbọchị.", "technology"),
    ("Ịntanetị na-enyere ndị mmadụ aka ịkparịta ụka.", "technology"),
    ("Onye mmemme na-ede koodu ọhụrụ.", "technology"),
    ("A na-echekwa data n'ime nchekwa kọmputa.", "technology"),
    ("Ụlọ ọrụ ahụ mepụtara ngwa ọhụrụ.", "technology"),
    ("Ọ na-amụ banyere ọgụgụ isi mmadụ.", "technology"),
    ("Ekwentị ahụ nwere batrị ọhụrụ.", "technology"),
    ("Ndị ọrụ na-eji ngwanrọ arụ ọrụ.", "technology"),
    ("Teknụzụ na-agbanwe ndụ mmadụ.", "technology"),
    ("Ọ na-amụta otú e si emepụta weebụsaịtị.", "technology"),
    ("Kọmputa ahụ anaghị arụ ọrụ nke ọma.", "technology"),
    ("Onye mmemme chọtara njehie na koodu ahụ.", "technology"),
    ("A na-eji paswọọdụ echekwa akaụntụ.", "technology"),
    ("Ọtụtụ mmadụ na-eji ngwa mkpanaka.", "technology"),
    ("Sistemụ ahụ na-arụ ọrụ ngwa ngwa.", "technology"),
    ("A na-eji nchekwa data echekwa ozi.", "technology"),
    ("Ọ mụtara asụsụ mmemme ọhụrụ.", "technology"),
    ("Teknụzụ nwere ike ime ka ọrụ dị mfe.", "technology"),
    ("Ndị ọrụ kwesịrị ichekwa ozi nkeonwe ha.", "technology"),
]


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "id",
            "igbo",
            "domain",
            "source"
        ])

        for index, (sentence, domain) in enumerate(
            sentences,
            start=1
        ):

            writer.writerow([
                f"MON{index:03d}",
                sentence,
                domain,
                "manual"
            ])

    print("=" * 60)
    print("MONOLINGUAL IGBO DATASET CREATED")
    print("=" * 60)
    print(f"Total sentences: {len(sentences)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nDomain distribution:")

    domains = {}

    for _, domain in sentences:
        domains[domain] = domains.get(domain, 0) + 1

    for domain, count in domains.items():
        print(f"{domain}: {count}")


if __name__ == "__main__":
    main()