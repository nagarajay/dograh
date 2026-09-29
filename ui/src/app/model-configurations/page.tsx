
import ModelConfigurationV2 from "@/components/ModelConfigurationV2";
import { RequireOrganization } from "@/components/RequireOrganization";
import { SETTINGS_DOCUMENTATION_URLS } from "@/constants/documentation";

export default function ServiceConfigurationPage() {
    return (
        <div className="min-h-screen">
            <div className="container mx-auto px-4 py-8">
                <div className="max-w-4xl mx-auto">
                    <RequireOrganization what="Model configuration">
                        <ModelConfigurationV2 docsUrl={SETTINGS_DOCUMENTATION_URLS.modelOverrides} />
                    </RequireOrganization>
                </div>
            </div>
        </div>
    );
}
