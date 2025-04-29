import { Component } from "@angular/core";

@Component({
    selector: "hlm-input-otp-fake-caret",
    standalone: true,
    template: `
        <div
            class="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div
                class="h-4 w-px animate-caret-blink bg-foreground duration-1000"></div>
        </div>
    `,
})
export class HlmInputOtpFakeCaretComponent {}
